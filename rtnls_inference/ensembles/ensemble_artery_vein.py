from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import networkx as nx
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

from .base import FundusEnsemble


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
    """Return defaults while ignoring legacy embedded refinement configuration."""
    del inference_config
    kwargs = {
        "vessel_threshold": 0.5,
        "crossing_threshold": 0.5,
        "crossing_radius": 3,
        "node_radius": 1,
        "min_component_size": 10,
        "relabel_margin": 0.15,
    }
    return "basic", kwargs


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


def _pixel_graph(skeleton: np.ndarray) -> nx.Graph:
    graph = nx.Graph()
    coordinates = np.argwhere(skeleton)
    graph.add_nodes_from(map(tuple, coordinates))
    for row, col in coordinates:
        node = (int(row), int(col))
        for drow, dcol in ((0, 1), (1, -1), (1, 0), (1, 1)):
            neighbour = (node[0] + drow, node[1] + dcol)
            if (
                0 <= neighbour[0] < skeleton.shape[0]
                and 0 <= neighbour[1] < skeleton.shape[1]
                and skeleton[neighbour]
            ):
                graph.add_edge(node, neighbour)
    return graph


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
    vessel_threshold: float = 0.5,
    crossing_threshold: float = 0.5,
    crossing_radius: int = 3,
    min_component_size: int = 10,
    relabel_margin: float = 0.15,
) -> np.ndarray:
    """Directionally reconnect segments across crossings, then relabel trees."""
    raw, vessel, crossing, skeleton, evidence_weight = _refinement_state(
        logits,
        vessel_threshold=vessel_threshold,
        crossing_threshold=crossing_threshold,
    )
    if int(vessel.sum()) < min_component_size:
        return raw

    try:
        crossing_zone = binary_dilation(crossing, disk(crossing_radius))
        component_image = _component_image(
            skeleton & ~crossing_zone,
            min_component_size=min_component_size,
        )
        if not np.any(component_image):
            return raw

        continuity = nx.Graph()
        continuity.add_nodes_from(range(1, int(component_image.max()) + 1))
        crossing_regions = connected_labels(crossing_zone, connectivity=2)
        for region_id in range(1, int(crossing_regions.max()) + 1):
            region = crossing_regions == region_id
            contacts = component_image[binary_dilation(region, disk(1))]
            contacts = [int(value) for value in np.unique(contacts) if value > 0]
            if len(contacts) < 2:
                continue
            center = np.asarray(np.argwhere(region).mean(axis=0))
            directions = {}
            nearby = binary_dilation(region, disk(2))
            for component_id in contacts:
                points = np.argwhere((component_image == component_id) & nearby)
                if len(points):
                    vector = points.mean(axis=0) - center
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        directions[component_id] = vector / norm
            remaining = set(directions)
            while len(remaining) >= 2:
                best = min(
                    (
                        (float(np.dot(directions[a], directions[b])), a, b)
                        for a in remaining
                        for b in remaining
                        if a < b
                    ),
                    default=None,
                )
                if best is None:
                    break
                _, first, second = best
                continuity.add_edge(first, second)
                remaining.remove(first)
                remaining.remove(second)

        group_image = np.zeros_like(component_image)
        for group_id, component_ids in enumerate(
            nx.connected_components(continuity), start=1
        ):
            for component_id in component_ids:
                group_image[component_image == component_id] = group_id

        return _relabel_groups(
            logits,
            raw,
            vessel,
            crossing,
            group_image,
            evidence_weight,
            min_component_size=min_component_size,
            relabel_margin=relabel_margin,
        )
    except Exception:  # noqa: BLE001 - inference must fall back to raw labels
        return raw


def refine_artery_vein_graph(logits: np.ndarray, **kwargs) -> np.ndarray:
    """Backward-compatible alias for full directional refinement."""
    return refine_artery_vein_graph_full(logits, **kwargs)


def artery_vein_refinement_variants(
    logits: np.ndarray,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Compute the basic, simple-segment, and full-directional masks."""
    basic_kwargs = {
        key: kwargs[key]
        for key in ("vessel_threshold", "crossing_threshold")
        if key in kwargs
    }
    full_kwargs = {key: value for key, value in kwargs.items() if key != "node_radius"}
    return {
        "basic": basic_artery_vein_mask(logits, **basic_kwargs),
        "refinement_simple": refine_artery_vein_segments(logits, **kwargs),
        "refinement_full": refine_artery_vein_graph_full(logits, **full_kwargs),
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
        return refine_artery_vein_segments(logits, **kwargs)
    full_kwargs = {key: value for key, value in kwargs.items() if key != "node_radius"}
    return refine_artery_vein_graph_full(logits, **full_kwargs)


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
        refinement_mode: str | None = None,
        refinement_parameters: Mapping[str, float | int] | None = None,
        **kwargs,
    ):
        """Create an AV ensemble with optional CPU graph refinement.

        Refinement is a runtime concern and is deliberately not read from the
        embedded model configuration. ``None`` selects basic per-pixel AV
        decoding without graph refinement.
        """
        super().__init__(*args, **kwargs)
        self.refinement_mode = normalize_refinement_mode(refinement_mode or "basic")
        _, defaults = resolve_artery_vein_refinement_config(None)
        parameters = dict(refinement_parameters or {})
        unknown = set(parameters) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown AV refinement parameters: {sorted(unknown)}")
        self.refinement_parameters = {**defaults, **parameters}

    def _halo_logits(self, image: torch.Tensor) -> torch.Tensor:
        inference = self.config.get("inference", {})
        model_config = self.config.get("lightningmodule", {})
        return halo_sliding_window_inference(
            image,
            self.ensemble,
            context_size=int(model_config.get("context_size", 1024)),
            output_size=int(model_config.get("output_size", 512)),
            overlap=float(inference.get("overlap", 0.5)),
            sw_batch_size=int(inference.get("batch_size", 1)),
            sigma_scale=float(inference.get("gaussian_sigma_scale", 0.125)),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits = self._halo_logits(image)
        inference = self.config.get("inference", {})
        if not inference.get("tta", False):
            return logits
        for axes in inference.get("tta_flips", [[2], [3], [2, 3]]):
            flipped = self._halo_logits(torch.flip(image, dims=axes))
            logits += torch.flip(flipped, dims=[axis + 1 for axis in axes])
        return logits / (len(inference.get("tta_flips", [[2], [3], [2, 3]])) + 1)

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
        if self.config.get("inference", {}).get(
            "return_postprocess_intermediates", False
        ):
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
