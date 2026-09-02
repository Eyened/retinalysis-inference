from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import Field, model_validator
from scipy import ndimage
from skimage.measure import label as connected_labels
from skimage.morphology import binary_dilation, disk, skeletonize
from tqdm import tqdm

from rtnls_inference.artery_vein import (
    AV_HEAD_NAMES,
)
from rtnls_inference.ensembles.predict_output import (
    PredictFullOutput,
    decollate_predict_full,
    output_manifest_row,
    require_rank,
    restore_array_to_preprocessed,
)

from .av_inference import AVInferenceConfig, infer_artery_vein
from .base import _UNSET, FundusEnsemble


def _tile_positions(length: int, output_size: int, stride: int) -> list[int]:
    canvas = max(length, output_size)
    positions = list(range(0, max(canvas - output_size, 0) + 1, stride))
    final = canvas - output_size
    if not positions or positions[-1] != final:
        positions.append(final)
    return positions


def gaussian_importance_map(
    output_size: int,
    *,
    sigma_scale: float = 0.125,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    coords = torch.arange(output_size, device=device, dtype=dtype)
    center = (output_size - 1) / 2.0
    sigma = max(output_size * sigma_scale, 1.0)
    one_dimensional = torch.exp(-0.5 * ((coords - center) / sigma) ** 2)
    importance = one_dimensional[:, None] * one_dimensional[None, :]
    return importance.clamp_min(torch.finfo(dtype).eps)


def halo_sliding_window_inference(
    image: torch.Tensor,
    predictor,
    *,
    context_size: int = 1024,
    output_size: int = 512,
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    sigma_scale: float = 0.125,
) -> torch.Tensor:
    """Stitch central halo predictions from a model or model ensemble."""
    if image.ndim != 4:
        raise ValueError(f"Expected NCHW input, got {image.shape}")
    if context_size <= output_size or (context_size - output_size) % 2:
        raise ValueError("context_size/output_size do not define a symmetric halo")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    stride = max(round(output_size * (1.0 - overlap)), 1)
    halo = (context_size - output_size) // 2
    importance = gaussian_importance_map(
        output_size,
        sigma_scale=sigma_scale,
        device=image.device,
        dtype=image.dtype,
    )

    sample_outputs: list[torch.Tensor] = []
    predictor_has_model_axis: bool | None = None
    for sample in image:
        height, width = sample.shape[-2:]
        canvas_h, canvas_w = max(height, output_size), max(width, output_size)
        bottom_extra, right_extra = canvas_h - height, canvas_w - width
        padded = F.pad(
            sample[None],
            (halo, halo + right_extra, halo, halo + bottom_extra),
            mode="constant",
            value=0.0,
        )[0]
        positions = [
            (top, left)
            for top in _tile_positions(canvas_h, output_size, stride)
            for left in _tile_positions(canvas_w, output_size, stride)
        ]

        accumulator = None
        normalizer = torch.zeros(
            (canvas_h, canvas_w), device=image.device, dtype=image.dtype
        )
        for start in range(0, len(positions), sw_batch_size):
            chunk_positions = positions[start : start + sw_batch_size]
            tiles = torch.stack(
                [
                    padded[
                        :,
                        top : top + context_size,
                        left : left + context_size,
                    ]
                    for top, left in chunk_positions
                ]
            )
            predictions = predictor(tiles)
            has_model_axis = predictions.ndim == 5
            if predictions.ndim == 4:
                predictions = predictions[None]
            elif predictions.ndim != 5:
                raise ValueError(
                    f"Halo predictor must return TCHW or MTCHW, got {predictions.shape}"
                )
            if predictor_has_model_axis is None:
                predictor_has_model_axis = has_model_axis
            elif predictor_has_model_axis != has_model_axis:
                raise RuntimeError("Predictor output rank changed between tile batches")
            if predictions.shape[-2:] != (output_size, output_size):
                raise ValueError(
                    f"Halo predictor returned {predictions.shape[-2:]}; "
                    f"expected {(output_size, output_size)}"
                )
            if accumulator is None:
                accumulator = torch.zeros(
                    (
                        predictions.shape[0],
                        predictions.shape[2],
                        canvas_h,
                        canvas_w,
                    ),
                    device=predictions.device,
                    dtype=predictions.dtype,
                )
            for tile_idx, (top, left) in enumerate(chunk_positions):
                accumulator[
                    ..., top : top + output_size, left : left + output_size
                ] += predictions[:, tile_idx] * importance
                normalizer[top : top + output_size, left : left + output_size] += (
                    importance
                )

        if accumulator is None or torch.any(normalizer <= 0):
            raise RuntimeError("Halo tiling left output pixels uncovered")
        sample_outputs.append(
            accumulator[..., :height, :width] / normalizer[:height, :width]
        )

    output = torch.stack(sample_outputs, dim=1)  # M,N,C,H,W
    if predictor_has_model_axis:
        return output.permute(1, 0, 2, 3, 4)  # N,M,C,H,W
    return output[0]  # N,C,H,W


AV_REFINEMENT_VARIANTS = (
    "basic",
    "refinement_simple",
    "refinement_full",
)

_REFINEMENT_MODE_ALIASES = {
    "none": "basic",
    "basic": "basic",
    "simple": "refinement_simple",
    "refinement_simple": "refinement_simple",
    "full": "refinement_full",
    "refinement_full": "refinement_full",
}


def normalize_refinement_mode(mode: str) -> str:
    try:
        return _REFINEMENT_MODE_ALIASES[mode.lower()]
    except KeyError as error:
        choices = ", ".join(sorted(_REFINEMENT_MODE_ALIASES))
        raise ValueError(
            f"Unknown AV refinement mode {mode!r}; choose from {choices}"
        ) from error


def resolve_artery_vein_refinement_config(
    inference_config: Mapping | None,
) -> tuple[str, dict[str, float | int]]:
    """Return structured AV inference defaults."""
    del inference_config
    return "basic", asdict(AVInferenceConfig())


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -80.0, 80.0)))


def _av_probabilities(logits: np.ndarray) -> np.ndarray:
    av_logits = logits[..., 1:3]
    shifted = av_logits - av_logits.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def basic_artery_vein_mask(
    logits: np.ndarray,
    *,
    vessel_threshold: float = 0.5,
    crossing_threshold: float = 0.5,
) -> np.ndarray:
    """Project windowed/TTA logits without graph-based refinement."""
    if logits.ndim != 3 or logits.shape[-1] < 4:
        raise ValueError(
            f"Expected HWC AV logits with at least four heads, got {logits.shape}"
        )
    vessel = _sigmoid(logits[..., 0]) >= vessel_threshold
    crossing = vessel & (_sigmoid(logits[..., 3]) >= crossing_threshold)
    result = np.zeros(vessel.shape, dtype=np.uint8)
    result[vessel] = np.where(logits[..., 1][vessel] >= logits[..., 2][vessel], 1, 2)
    result[crossing] = 3
    return result


def _component_image(
    binary: np.ndarray,
    *,
    min_component_size: int,
) -> np.ndarray:
    labels = connected_labels(binary, connectivity=2)
    result = np.zeros(binary.shape, dtype=np.int32)
    next_id = 1
    for component_id in range(1, int(labels.max()) + 1):
        pixels = labels == component_id
        if int(pixels.sum()) < min_component_size:
            continue
        result[pixels] = next_id
        next_id += 1
    return result


def _refinement_state(
    logits: np.ndarray,
    *,
    vessel_threshold: float,
    crossing_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = basic_artery_vein_mask(
        logits,
        vessel_threshold=vessel_threshold,
        crossing_threshold=crossing_threshold,
    )
    vessel_probability = _sigmoid(logits[..., 0])
    vessel = vessel_probability >= vessel_threshold
    crossing = vessel & (_sigmoid(logits[..., 3]) >= crossing_threshold)
    av_probability = _av_probabilities(logits)
    evidence_weight = vessel_probability * np.abs(
        av_probability[..., 0] - av_probability[..., 1]
    )
    return raw, vessel, crossing, skeletonize(vessel), evidence_weight


def _relabel_groups(
    logits: np.ndarray,
    raw: np.ndarray,
    vessel: np.ndarray,
    crossing: np.ndarray,
    group_image: np.ndarray,
    evidence_weight: np.ndarray,
    *,
    min_component_size: int,
    relabel_margin: float,
) -> np.ndarray:
    assigned: dict[int, int] = {}
    av_log_odds = logits[..., 1] - logits[..., 2]
    for group_id in range(1, int(group_image.max()) + 1):
        pixels = group_image == group_id
        if int(pixels.sum()) < min_component_size:
            continue
        weights = evidence_weight[pixels]
        weight_sum = float(weights.sum())
        if weight_sum <= np.finfo(np.float32).eps:
            continue
        score = float(np.sum(av_log_odds[pixels] * weights) / weight_sum)
        if abs(score) >= relabel_margin:
            assigned[group_id] = 1 if score > 0 else 2

    if not assigned:
        return raw
    _, nearest = ndimage.distance_transform_edt(group_image == 0, return_indices=True)
    nearest_groups = group_image[nearest[0], nearest[1]]
    refined = raw.copy()
    for group_id, class_id in assigned.items():
        pixels = vessel & ~crossing & (nearest_groups == group_id)
        refined[pixels] = class_id
    refined[~vessel] = 0
    refined[crossing] = 3
    return refined


def refine_artery_vein_segments(
    logits: np.ndarray,
    *,
    vessel_threshold: float = 0.5,
    crossing_threshold: float = 0.5,
    crossing_radius: int = 3,
    node_radius: int = 1,
    min_component_size: int = 10,
    relabel_margin: float = 0.15,
) -> np.ndarray:
    """Relabel individual skeleton segments cut at graph nodes."""
    raw, vessel, crossing, skeleton, evidence_weight = _refinement_state(
        logits,
        vessel_threshold=vessel_threshold,
        crossing_threshold=crossing_threshold,
    )
    if int(vessel.sum()) < min_component_size:
        return raw

    try:
        kernel = np.ones((3, 3), dtype=np.uint8)
        neighbour_count = ndimage.convolve(
            skeleton.astype(np.uint8), kernel, mode="constant", cval=0
        ) - skeleton.astype(np.uint8)
        graph_nodes = skeleton & (neighbour_count > 2)
        cut_zone = binary_dilation(graph_nodes, disk(node_radius))
        cut_zone |= binary_dilation(crossing, disk(crossing_radius))
        segment_image = _component_image(
            skeleton & ~cut_zone,
            min_component_size=min_component_size,
        )
        if not np.any(segment_image):
            return raw
        return _relabel_groups(
            logits,
            raw,
            vessel,
            crossing,
            segment_image,
            evidence_weight,
            min_component_size=min_component_size,
            relabel_margin=relabel_margin,
        )
    except Exception:  # noqa: BLE001 - inference must fall back to raw labels
        return raw


def refine_artery_vein_graph_full(
    logits: np.ndarray,
    *,
    disc_mask: np.ndarray | None = None,
    bounds_mask: np.ndarray | None = None,
    **kwargs: float,
) -> np.ndarray:
    """Infer anatomically rooted acyclic artery and vein forests."""
    return infer_artery_vein(
        logits, disc_mask=disc_mask, bounds_mask=bounds_mask, **kwargs
    ).mask


def refine_artery_vein_graph_full_with_flips(
    logits: np.ndarray,
    *,
    disc_mask: np.ndarray | None = None,
    bounds_mask: np.ndarray | None = None,
    **kwargs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return structured refinement and segments whose AV label changed."""
    result = infer_artery_vein(
        logits, disc_mask=disc_mask, bounds_mask=bounds_mask, **kwargs
    )
    return result.mask, result.changed_segment_mask


def refine_artery_vein_graph(logits: np.ndarray, **kwargs) -> np.ndarray:
    """Backward-compatible alias for structured AV forest inference."""
    return refine_artery_vein_graph_full(logits, **kwargs)


def _simple_refinement_kwargs(kwargs: Mapping) -> dict[str, float | int]:
    """Map structured settings onto the legacy simple-segment decoder."""
    junction_radius = int(kwargs.get("junction_radius", 3))
    return {
        "vessel_threshold": float(kwargs.get("vessel_threshold", 0.5)),
        "crossing_threshold": float(kwargs.get("crossing_threshold", 0.5)),
        "crossing_radius": junction_radius,
        "node_radius": junction_radius,
        "min_component_size": int(kwargs.get("min_segment_size", 10)),
        "relabel_margin": 0.15,
    }


def artery_vein_refinement_variants(
    logits: np.ndarray,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Compute basic, simple-segment, and structured-forest AV masks."""
    basic_kwargs = {
        key: kwargs[key]
        for key in ("vessel_threshold", "crossing_threshold")
        if key in kwargs
    }
    return {
        "basic": basic_artery_vein_mask(logits, **basic_kwargs),
        "refinement_simple": refine_artery_vein_segments(
            logits, **_simple_refinement_kwargs(kwargs)
        ),
        "refinement_full": refine_artery_vein_graph_full(logits, **kwargs),
    }


def refine_artery_vein(
    logits: np.ndarray,
    mode: str,
    **kwargs,
) -> np.ndarray:
    """Apply one configured refinement mode without computing unused variants."""
    normalized = normalize_refinement_mode(mode)
    if normalized == "basic":
        basic_kwargs = {
            key: kwargs[key]
            for key in ("vessel_threshold", "crossing_threshold")
            if key in kwargs
        }
        return basic_artery_vein_mask(logits, **basic_kwargs)
    if normalized == "refinement_simple":
        return refine_artery_vein_segments(logits, **_simple_refinement_kwargs(kwargs))
    return refine_artery_vein_graph_full(logits, **kwargs)


class ArteryVeinPredictFull(PredictFullOutput):
    logits: np.ndarray = Field(
        description="Ensemble-averaged four-head logits as NHW4."
    )
    logit_names: tuple[str, ...] = Field(
        description="Names corresponding to the final logits axis."
    )
    refinement_mode: str = Field(
        description="Configured CPU artery/vein refinement mode."
    )
    refinement_parameters: dict[str, float | int] = Field(
        description="Parameters used for per-item artery/vein refinement."
    )

    @model_validator(mode="after")
    def validate_artery_vein(self):
        require_rank(self.prediction, 5, "prediction")
        require_rank(self.logits, 4, "logits")
        require_rank(self.aggregate, 4, "aggregate")
        if self.prediction.shape[-1] != len(AV_HEAD_NAMES) or self.logits.shape[
            -1
        ] != len(AV_HEAD_NAMES):
            raise ValueError(
                f"AV member and averaged logits must have {len(AV_HEAD_NAMES)} heads"
            )
        if self.aggregate.shape[-1] != 4:
            raise ValueError("AV aggregate must contain four class probabilities")
        if self.prediction.shape[2:4] != self.logits.shape[1:3]:
            raise ValueError("AV member and averaged logits spatial sizes differ")
        if self.aggregate.shape[:3] != self.logits.shape[:3]:
            raise ValueError("AV probabilities and logits spatial sizes differ")
        if tuple(self.logit_names) != tuple(AV_HEAD_NAMES):
            raise ValueError("AV logit_names do not match AV_HEAD_NAMES")
        if not all(
            np.issubdtype(value.dtype, np.floating)
            for value in (self.prediction, self.aggregate, self.logits)
        ):
            raise ValueError("AV predictions and logits must be floating point")
        return self


class ArteryVeinSegmentationEnsemble(FundusEnsemble):
    """Halo-tile AV inference, native head export, and graph refinement."""

    predict_full_model = ArteryVeinPredictFull
    autocast_inference = True

    def __init__(
        self,
        *args,
        refinement_mode: str | None = _UNSET,
        refinement_parameters: Mapping[str, float | int] | None = _UNSET,
        **kwargs,
    ):
        """Create an AV ensemble with optional CPU graph refinement.

        Embedded ``inference.graph_refinement`` is the default. Constructor
        ``graph_refinement`` merges over it. ``refinement_mode`` and
        ``refinement_parameters`` remain higher-precedence aliases; passing
        ``refinement_mode=None`` still selects basic per-pixel decoding.
        """
        super().__init__(*args, **kwargs)
        _, defaults = resolve_artery_vein_refinement_config(None)
        merged = self._inference_mapping("graph_refinement")
        if refinement_mode is not _UNSET:
            mode = refinement_mode
        else:
            mode = merged.get("mode")
        self.refinement_mode = normalize_refinement_mode(mode or "basic")
        parameters = {key: value for key, value in merged.items() if key != "mode"}
        if refinement_parameters is not _UNSET and refinement_parameters is not None:
            parameters.update(refinement_parameters)
        unknown = set(parameters) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown AV refinement parameters: {sorted(unknown)}")
        self.refinement_parameters = {**defaults, **parameters}

    def _halo_logits(self, image: torch.Tensor) -> torch.Tensor:
        model_config = self.config.get("lightningmodule", {})
        return halo_sliding_window_inference(
            image,
            self.ensemble,
            context_size=int(model_config.get("context_size", 1024)),
            output_size=int(model_config.get("output_size", 512)),
            overlap=float(self._inference_setting("overlap", 0.5)),
            sw_batch_size=self._tile_batch_size(1, legacy_batch_size=True),
            sigma_scale=float(self._inference_setting("gaussian_sigma_scale", 0.125)),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits = self._halo_logits(image)
        if not self._inference_setting("tta", False):
            return logits
        flip_axes = self._inference_setting("tta_flips", [[2], [3], [2, 3]])
        for axes in flip_axes:
            flipped = self._halo_logits(torch.flip(image, dims=axes))
            logits += torch.flip(flipped, dims=[axis + 1 for axis in axes])
        return logits / (len(flip_axes) + 1)

    def _refinement_config(self) -> tuple[str, dict]:
        return self.refinement_mode, dict(self.refinement_parameters)

    def _predict_member_tensors(self, batch):
        return self.forward(batch["image"]).permute(0, 1, 3, 4, 2)

    def _aggregate_tensors(self, member_output):
        logits = member_output.mean(dim=1)
        head_probabilities = torch.sigmoid(logits)
        vessel = head_probabilities[..., 0]
        crossing = head_probabilities[..., 3]
        av_fraction = torch.softmax(logits[..., 1:3], dim=-1)
        background = 1.0 - vessel
        crossing_class = vessel * crossing
        remaining = vessel * (1.0 - crossing)
        return torch.stack(
            [
                background,
                remaining * av_fraction[..., 0],
                remaining * av_fraction[..., 1],
                crossing_class,
            ],
            dim=-1,
        )

    def _predict_step_tensors(self, batch):
        prediction = self._predict_member_tensors(batch)
        logits = prediction.mean(dim=1)
        aggregate = self._aggregate_tensors(prediction)
        refinement_mode, refinement_kwargs = self._refinement_config()
        return {
            "prediction": prediction,
            "aggregate": aggregate,
            "logits": logits,
            "logit_names": tuple(AV_HEAD_NAMES),
            "refinement_mode": refinement_mode,
            "refinement_parameters": refinement_kwargs,
        }

    def _prediction_size(self, batch, output):
        aggregate = output["aggregate"]
        return int(aggregate.shape[1]), int(aggregate.shape[2])

    def postprocess_item(self, item):
        refined = refine_artery_vein(
            np.asarray(item["logits"]),
            item["refinement_mode"],
            **item["refinement_parameters"],
        )
        restored_mask = restore_array_to_preprocessed(
            refined, item["geometry"], "nearest"
        ).astype(np.uint8, copy=False)
        result = {
            "id": item.get("id"),
            "output": restored_mask,
            "refined_mask": restored_mask,
            "output_kind": "artery_vein_mask",
            "output_space": "preprocessed",
            "geometry": item["geometry"],
            "refinement_mode": item["refinement_mode"],
        }
        if "preprocessed_image" in item:
            result["preprocessed_image"] = item["preprocessed_image"]
        if self._inference_setting("return_postprocess_intermediates", False):
            result["probabilities"] = restore_array_to_preprocessed(
                item["aggregate"], item["geometry"], "bilinear"
            )
            result["logits"] = restore_array_to_preprocessed(
                item["logits"], item["geometry"], "bilinear"
            )
            result["logit_names"] = item["logit_names"]
        return result

    def _compatibility_items(self, full_output):
        items = []
        for item in decollate_predict_full(full_output):
            refined = refine_artery_vein(
                item["logits"],
                item["refinement_mode"],
                **item["refinement_parameters"],
            )
            head_probabilities = 1.0 / (1.0 + np.exp(-item["logits"]))
            items.append(
                {
                    "id": item.get("id"),
                    "image": item["aggregate"],
                    "heads": {
                        name: head_probabilities[..., index]
                        for index, name in enumerate(AV_HEAD_NAMES)
                    },
                    "head_logits": item["logits"],
                    "refinement_mode": item["refinement_mode"],
                    "refined_mask": refined,
                }
            )
        return items

    def _predict_output_batch(self, batch: dict) -> list[dict]:
        return [
            self.postprocess_item(item)
            for item in decollate_predict_full(self.predict_step_full(batch))
        ]

    @staticmethod
    def _save_item(item: dict, dest_path: str | Path):
        Image.fromarray(item["refined_mask"].astype(np.uint8)).save(dest_path)

    def _predict_dataloader(self, dataloader, dest_path):
        if dest_path is None:
            raise ValueError("dest_path is required for spatial prediction output")
        os.makedirs(dest_path, exist_ok=True)
        manifest = []
        for batch in tqdm(dataloader):
            if not batch:
                continue
            for item in self._predict_output_batch(batch):
                fpath = Path(dest_path) / f"{item['id']}.png"
                self._save_item(item, fpath)
                manifest.append(output_manifest_row(item, str(fpath)))
        return pd.DataFrame(manifest)
