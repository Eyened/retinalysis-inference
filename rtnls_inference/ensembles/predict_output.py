from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any, Literal

import cv2
import numpy as np
import torch

from rtnls_inference.inputs import batch_ids


def require_tensor(value, field_name, *, rank=None, floating=False):
    """Check backend tensors without copying data or synchronizing the device."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"{field_name} must be a torch.Tensor, got {type(value).__name__}"
        )
    if rank is not None and value.ndim != rank:
        raise ValueError(
            f"{field_name} must have rank {rank}, got shape {tuple(value.shape)}"
        )
    if any(size == 0 for size in value.shape):
        raise ValueError(
            f"{field_name} dimensions, including batch/member axes, must be positive, got {tuple(value.shape)}"
        )
    if floating and not value.is_floating_point():
        raise TypeError(f"{field_name} must be floating point, got {value.dtype}")


def check_prediction_tensors(output, batch_size):
    """Check shared batch/member axes and optional spatial intermediates once."""
    prediction, aggregate = output["prediction"], output["aggregate"]
    require_tensor(prediction, "prediction")
    require_tensor(aggregate, "aggregate")
    if prediction.ndim < 2 or prediction.shape[1] < 1:
        raise ValueError(
            f"prediction must have non-empty batch/member axes, got {tuple(prediction.shape)}"
        )
    if (
        aggregate.ndim < 1
        or prediction.shape[0] != batch_size
        or aggregate.shape[0] != batch_size
    ):
        raise ValueError(
            f"Prediction and input batch sizes differ: expected {batch_size}, "
            f"prediction={tuple(prediction.shape)}, aggregate={tuple(aggregate.shape)}"
        )
    if aggregate.ndim != prediction.ndim - 1:
        raise ValueError(
            f"aggregate must have rank {prediction.ndim - 1}, got {tuple(aggregate.shape)}"
        )
    if "logits" in output:
        logits = output["logits"]
        require_tensor(logits, "logits", rank=aggregate.ndim, floating=True)
        expected = (batch_size, *prediction.shape[2:])
        if logits.shape != expected:
            raise ValueError(
                f"logits must have shape {expected}, got {tuple(logits.shape)}"
            )
        # Head logits may project to a different number of output classes.
        if logits.shape[:-1] != aggregate.shape[:-1]:
            raise ValueError(
                f"logits and aggregate spatial shapes differ: {tuple(logits.shape)} and {tuple(aggregate.shape)}"
            )
    elif prediction.shape[2:] != aggregate.shape[1:]:
        raise ValueError(
            f"prediction and aggregate output shapes differ: {tuple(prediction.shape)} and {tuple(aggregate.shape)}"
        )
    if "heatmaps" in output:
        heatmaps = output["heatmaps"]
        require_tensor(heatmaps, "heatmaps", rank=5)
        if heatmaps.shape[:3] != prediction.shape[:3]:
            raise ValueError(
                f"heatmaps and prediction batch/member/keypoint axes differ: {tuple(heatmaps.shape)} and {tuple(prediction.shape)}"
            )


def _spatial_size(value, field_name):
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(
            isinstance(size, bool) or not isinstance(size, Integral) or size <= 0
            for size in value
        )
    ):
        raise ValueError(
            f"{field_name} must contain two positive integer dimensions, got {value!r}"
        )
    return tuple(map(int, value))


def _geometry_dict(geometry: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(geometry, Mapping):
        raise TypeError("geometry must be a mapping")
    result = {
        "preprocessed_size": _spatial_size(
            geometry.get("preprocessed_size"), "preprocessed_size"
        )
    }
    if geometry.get("prediction_size") is not None:
        result["prediction_size"] = _spatial_size(
            geometry["prediction_size"], "prediction_size"
        )
    return result


def to_numpy(value: Any) -> Any:
    """Recursively detach tensors without changing ordinary metadata."""
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.cpu().numpy()
    if isinstance(value, Mapping):
        return {key: to_numpy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(to_numpy(item) for item in value)
    if isinstance(value, list):
        return [to_numpy(item) for item in value]
    return value


def model_input_size(batch: Mapping[str, Any]) -> tuple[int, int]:
    image = batch["image"]
    if not isinstance(image, (torch.Tensor, np.ndarray)) or image.ndim != 4:
        raise ValueError("batch['image'] must be an NCHW tensor or array")
    return int(image.shape[-2]), int(image.shape[-1])


def canonical_context(
    batch: Mapping[str, Any],
    *,
    batch_size: int,
    prediction_size: tuple[int, int] | None,
) -> dict[str, Any]:
    """Extract canonical CPU context without moving it to the prediction device."""
    result: dict[str, Any] = {}
    if "id" in batch:
        result["id"] = batch_ids(batch)

    canonical = batch.get("preprocessed_image")
    if canonical is not None:
        canonical = np.asarray(to_numpy(canonical))
        if canonical.ndim == 4 and canonical.shape[1] == 3 and canonical.shape[-1] != 3:
            canonical = np.moveaxis(canonical, 1, -1)
        if np.issubdtype(canonical.dtype, np.floating):
            maximum = float(canonical.max(initial=0))
            if maximum <= 1.0:
                canonical = canonical * 255.0
            canonical = np.clip(canonical, 0, 255).astype(np.uint8)
        else:
            canonical = canonical.astype(np.uint8, copy=False)
        if canonical.ndim != 4 or canonical.shape[-1] != 3:
            raise ValueError(
                f"preprocessed_image must have shape NHWC with 3 channels, got {canonical.shape}"
            )
        if canonical.shape[0] != batch_size:
            raise ValueError(
                f"preprocessed_image batch size must be {batch_size}, got {canonical.shape[0]}"
            )
        result["preprocessed_image"] = canonical
        sizes = [(int(image.shape[0]), int(image.shape[1])) for image in canonical]
    else:
        sizes = [model_input_size(batch)] * batch_size

    result["geometry"] = [
        _geometry_dict({"preprocessed_size": size, "prediction_size": prediction_size})
        for size in sizes
    ]
    return result


def restore_array_to_preprocessed(
    array: np.ndarray,
    geometry: Mapping[str, Any],
    interpolation: Literal["bilinear", "nearest"] = "bilinear",
) -> np.ndarray:
    """Resize an HW/HWC array to canonical preprocessing space."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"array must be a numpy.ndarray, got {type(array).__name__}")
    value = array
    if value.ndim not in (2, 3):
        raise ValueError(f"Expected HW or HWC array, got {value.shape}")
    if any(size == 0 for size in value.shape):
        raise ValueError(f"array dimensions must be positive, got {value.shape}")
    info = _geometry_dict(geometry)
    target_h, target_w = map(int, info["preprocessed_size"])
    prediction_size = info.get("prediction_size")
    if (
        prediction_size is not None
        and tuple(map(int, prediction_size)) != value.shape[:2]
    ):
        raise ValueError(
            f"Array size {value.shape[:2]} differs from prediction_size {prediction_size}"
        )
    if value.shape[:2] == (target_h, target_w):
        return value.copy()
    mode = {"bilinear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST}[interpolation]
    cv_value = value.astype(np.float32) if value.dtype == np.float16 else value
    restored = cv2.resize(cv_value, (target_w, target_h), interpolation=mode)
    if value.ndim == 3 and restored.ndim == 2:
        restored = restored[..., None]
    return restored.astype(value.dtype, copy=False)


def restore_points_to_preprocessed(
    points: np.ndarray,
    geometry: Mapping[str, Any],
) -> np.ndarray:
    """Scale xy points from prediction space to canonical preprocessing space."""
    if not isinstance(points, np.ndarray):
        raise TypeError(f"points must be a numpy.ndarray, got {type(points).__name__}")
    value = points
    if value.ndim < 1 or value.shape[-1] != 2:
        raise ValueError(f"Expected points with final xy axis, got {value.shape}")
    info = _geometry_dict(geometry)
    prediction_size = info.get("prediction_size")
    if prediction_size is None:
        return value.copy()
    prediction_h, prediction_w = map(float, prediction_size)
    target_h, target_w = map(float, info["preprocessed_size"])
    restored = value.astype(np.result_type(value.dtype, np.float32), copy=True)
    restored[..., 0] *= target_w / prediction_w
    restored[..., 1] *= target_h / prediction_h
    return restored


def unbatch_predictions(
    predictions: Mapping[str, Any], batch: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return raw per-image NumPy predictions with IDs and canonical geometry.

    Only top-level prediction arrays have a batch axis. Shared metadata, such
    as refinement configurations or head names, is never split by its length.
    No resizing or task postprocessing takes place, and inputs are not mutated.
    """
    if not isinstance(predictions, Mapping):
        raise TypeError(
            f"predictions must be a mapping, got {type(predictions).__name__}"
        )
    for name in ("aggregate", "prediction_size"):
        if name not in predictions:
            raise ValueError(f"Missing prediction field {name!r}")
    model_input_size(batch)
    batch_size = int(batch["image"].shape[0])
    aggregate = predictions["aggregate"]
    if not isinstance(aggregate, (torch.Tensor, np.ndarray)):
        raise TypeError(
            f"aggregate must be a tensor or NumPy array, got {type(aggregate).__name__}"
        )
    if aggregate.ndim < 1 or aggregate.shape[0] != batch_size:
        raise ValueError(
            f"Prediction and input batch sizes differ: aggregate expected batch axis "
            f"{batch_size}, got shape {tuple(aggregate.shape)}"
        )
    context = canonical_context(
        batch, batch_size=batch_size, prediction_size=predictions["prediction_size"]
    )
    for name, value in context.items():
        if len(value) != batch_size:
            raise ValueError(
                f"{name} must have {batch_size} batch items, got {len(value)}"
            )
    output = to_numpy(
        {
            name: value
            for name, value in predictions.items()
            if name != "prediction_size"
        }
    )
    for name, value in output.items():
        if isinstance(value, np.ndarray) and (
            value.ndim < 1 or value.shape[0] != batch_size
        ):
            raise ValueError(
                f"Prediction field {name!r} must have batch axis {batch_size}, got shape {value.shape}"
            )
    return [
        {
            **{
                name: value[index] if isinstance(value, np.ndarray) else value
                for name, value in output.items()
            },
            **{name: value[index] for name, value in context.items()},
        }
        for index in range(batch_size)
    ]


def output_manifest_row(
    item: Mapping[str, Any], output_path: str | None
) -> dict[str, Any]:
    geometry = _geometry_dict(item["geometry"])
    prep_h, prep_w = geometry["preprocessed_size"]
    prediction_size = geometry.get("prediction_size")
    prediction_h, prediction_w = prediction_size or (None, None)
    return {
        "id": item.get("id"),
        "output_path": output_path,
        "output_kind": item["output_kind"],
        "output_space": "preprocessed",
        "preprocessed_height": prep_h,
        "preprocessed_width": prep_w,
        "prediction_height": prediction_h,
        "prediction_width": prediction_w,
    }
