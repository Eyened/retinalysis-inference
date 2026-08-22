from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning.utilities import move_data_to_device
from scipy import ndimage
from skimage.measure import label as connected_labels
from skimage.morphology import binary_dilation, disk, skeletonize
from tqdm import tqdm

from rtnls_inference.artery_vein import (
    AV_HEAD_NAMES,
    save_av_head_logits,
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
    """Resolve the shared AV refinement mode and parameters from inference config."""
    graph_config = (inference_config or {}).get("graph_refinement", {})
    mode = normalize_refinement_mode(str(graph_config.get("mode", "full")))
    kwargs = {
        "vessel_threshold": float(graph_config.get("vessel_threshold", 0.5)),
        "crossing_threshold": float(graph_config.get("crossing_threshold", 0.5)),
        "crossing_radius": int(graph_config.get("crossing_radius", 3)),
        "node_radius": int(graph_config.get("node_radius", 1)),
        "min_component_size": int(graph_config.get("min_component_size", 10)),
        "relabel_margin": float(graph_config.get("relabel_margin", 0.15)),
    }
    return mode, kwargs


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


class ArteryVeinSegmentationEnsemble(FundusEnsemble):
    """Halo-tile AV inference, native head export, and graph refinement."""

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

    def predict_head_logits_step(self, batch, batch_idx=None) -> torch.Tensor:
        return self.forward(batch["image"]).mean(dim=1).permute(0, 2, 3, 1)

    def _refinement_config(self) -> tuple[str, dict]:
        return resolve_artery_vein_refinement_config(self.config.get("inference"))

    def predict_step(self, batch, batch_idx=None):
        logits = self.predict_head_logits_step(batch, batch_idx)
        heads = torch.sigmoid(logits)
        vessel = heads[..., 0]
        crossing = heads[..., 3]
        av_fraction = torch.softmax(logits[..., 1:3], dim=-1)
        background = 1.0 - vessel
        crossing_class = vessel * crossing
        remaining = vessel * (1.0 - crossing)
        legacy = torch.stack(
            [
                background,
                remaining * av_fraction[..., 0],
                remaining * av_fraction[..., 1],
                crossing_class,
            ],
            dim=-1,
        )
        refinement_mode, refinement_kwargs = self._refinement_config()
        selected_masks = [
            refine_artery_vein(
                sample.detach().cpu().numpy(),
                refinement_mode,
                **refinement_kwargs,
            )
            for sample in logits
        ]
        return {
            "image": legacy,
            "heads": {name: heads[..., idx] for idx, name in enumerate(AV_HEAD_NAMES)},
            "head_logits": logits,
            "refinement_mode": refinement_mode,
            "refined_mask": torch.as_tensor(
                np.stack(selected_masks), device=logits.device, dtype=torch.uint8
            ),
        }

    def _predict_output_batch(self, batch: dict) -> list[dict]:
        with torch.autocast(device_type=self.get_device().type):
            device_batch = move_data_to_device(batch, self.get_device())
            output = self.predict_step(device_batch)
        items = []
        for index, identifier in enumerate(batch["id"]):
            item = {
                "id": identifier,
                "image": output["image"][index].detach().cpu().numpy(),
                "heads": {
                    name: value[index].detach().cpu().numpy()
                    for name, value in output["heads"].items()
                },
                "head_logits": output["head_logits"][index].detach().cpu().numpy(),
                "refinement_mode": output["refinement_mode"],
                "refined_mask": output["refined_mask"][index].detach().cpu().numpy(),
            }
            if "metadata" in batch:
                item["metadata"] = batch["metadata"][index]
            items.append(item)
        return items

    @staticmethod
    def _save_item(item: dict, dest_path: str | Path):
        Image.fromarray(item["refined_mask"].astype(np.uint8)).save(dest_path)

    @staticmethod
    def _undo_array(item: dict, value: np.ndarray, transform) -> np.ndarray:
        payload = {"image": value}
        if "metadata" in item:
            payload["metadata"] = item["metadata"]
        return transform.undo_item(payload)["image"]

    def _predict_dataloader(self, dataloader, dest_path):
        os.makedirs(dest_path, exist_ok=True)
        with torch.no_grad():
            for batch in tqdm(dataloader):
                if not batch:
                    continue
                for item in self._predict_output_batch(batch):
                    one_hot = np.eye(4, dtype=np.float32)[item["refined_mask"]]
                    item["refined_mask"] = np.argmax(
                        self._undo_array(item, one_hot, dataloader.dataset.transform),
                        axis=-1,
                    ).astype(np.uint8)
                    self._save_item(item, Path(dest_path) / f"{item['id']}.png")

    def _predict_head_logits_dataloader(
        self,
        dataloader,
        dest_path: str | Path,
        dtype: np.dtype | type = np.float16,
    ):
        os.makedirs(dest_path, exist_ok=True)
        with torch.no_grad():
            for batch in tqdm(dataloader):
                if not batch:
                    continue
                for item in self._predict_output_batch(batch):
                    logits = self._undo_array(
                        item,
                        item["head_logits"],
                        dataloader.dataset.transform,
                    )
                    save_av_head_logits(
                        Path(dest_path) / f"{item['id']}.npz",
                        logits,
                        dtype=dtype,
                    )

    def predict_head_logits_dataset(
        self, data, dest_path, num_workers=0, batch_size=None, dtype=np.float16
    ):
        dataloader = self._make_inference_dataloader(
            {"images": data},
            num_workers=num_workers,
            preprocess=True,
            batch_size=batch_size,
        )
        return self._predict_head_logits_dataloader(dataloader, dest_path, dtype)

    def predict_head_logits_preprocessed(
        self, data, dest_path, num_workers=0, batch_size=None, dtype=np.float16
    ):
        dataloader = self._make_inference_dataloader(
            {"images": data},
            num_workers=num_workers,
            preprocess=False,
            batch_size=batch_size,
        )
        return self._predict_head_logits_dataloader(dataloader, dest_path, dtype)
