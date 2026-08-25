from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import cv2
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PredictionGeometry(BaseModel):
    """Resize-only geometry between prediction and canonical preprocessing space."""

    model_config = ConfigDict(extra="forbid")

    preprocessed_size: tuple[int, int] = Field(
        description="Canonical preprocessed image size as (height, width)."
    )
    prediction_size: tuple[int, int] | None = Field(
        default=None,
        description="Spatial prediction size as (height, width), when applicable.",
    )

    @model_validator(mode="after")
    def validate_sizes(self):
        if any(size <= 0 for size in self.preprocessed_size):
            raise ValueError("preprocessed_size dimensions must be positive")
        if self.prediction_size is not None and any(
            size <= 0 for size in self.prediction_size
        ):
            raise ValueError("prediction_size dimensions must be positive")
        return self


class PredictFullOutput(BaseModel):
    """Validated, batch-major NumPy representation of one inference pass."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    prediction: np.ndarray = Field(
        description="Per-member predictions with batch and member axes first (NM...)."
    )
    aggregate: np.ndarray = Field(
        description="Member-aggregated, tensor-safe prediction with batch axis first (N...)."
    )
    preprocessed_image: np.ndarray | None = Field(
        default=None,
        description="Canonical RGB images as uint8 NHWC, before model-specific transforms.",
    )
    geometry: list[PredictionGeometry] | None = Field(
        default=None,
        description="Per-item resize-only canonical prediction geometry.",
    )

    @model_validator(mode="after")
    def validate_batch_axes(self):
        if self.prediction.ndim < 2:
            raise ValueError("prediction must have batch and member axes")
        if self.aggregate.ndim < 1:
            raise ValueError("aggregate must have a batch axis")
        batch_size = self.aggregate.shape[0]
        if self.prediction.shape[0] != batch_size:
            raise ValueError("prediction and aggregate batch sizes differ")
        if self.prediction.shape[1] < 1:
            raise ValueError("prediction must contain at least one ensemble member")
        if self.preprocessed_image is not None:
            image = self.preprocessed_image
            if image.ndim != 4 or image.shape[-1] != 3:
                raise ValueError(
                    "preprocessed_image must have shape NHWC with 3 channels"
                )
            if image.shape[0] != batch_size:
                raise ValueError("preprocessed_image batch size differs from aggregate")
            if image.dtype != np.uint8:
                raise ValueError("preprocessed_image must have dtype uint8")
        if self.geometry is not None and len(self.geometry) != batch_size:
            raise ValueError("geometry length differs from aggregate batch size")
        extra = self.__pydantic_extra__ or {}
        if "id" in extra and len(extra["id"]) != batch_size:
            raise ValueError("id length differs from aggregate batch size")
        return self


def require_rank(value: np.ndarray, rank: int, field_name: str) -> None:
    if value.ndim != rank:
        raise ValueError(f"{field_name} must have rank {rank}, got {value.shape}")


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


def validate_predict_full(
    model: type[PredictFullOutput], output: Mapping[str, Any]
) -> dict[str, Any]:
    validated = model.model_validate(dict(output))
    return validated.model_dump(mode="python", exclude_none=True)


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
        identifiers = to_numpy(batch["id"])
        if isinstance(identifiers, np.ndarray):
            identifiers = identifiers.tolist()
        elif isinstance(identifiers, Sequence) and not isinstance(identifiers, str):
            identifiers = list(identifiers)
        elif batch_size == 1:
            identifiers = [identifiers]
        result["id"] = identifiers

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
        result["preprocessed_image"] = canonical
        sizes = [(int(image.shape[0]), int(image.shape[1])) for image in canonical]
    else:
        sizes = [model_input_size(batch)] * batch_size

    result["geometry"] = [
        PredictionGeometry(preprocessed_size=size, prediction_size=prediction_size)
        for size in sizes
    ]
    return result


def _geometry_dict(geometry: PredictionGeometry | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(geometry, PredictionGeometry):
        return geometry.model_dump(mode="python")
    return dict(geometry)


def restore_array_to_preprocessed(
    array: np.ndarray,
    geometry: PredictionGeometry | Mapping[str, Any],
    interpolation: Literal["bilinear", "nearest"] = "bilinear",
) -> np.ndarray:
    """Resize an HW/HWC array to canonical preprocessing space."""
    value = np.asarray(array)
    if value.ndim not in (2, 3):
        raise ValueError(f"Expected HW or HWC array, got {value.shape}")
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
    geometry: PredictionGeometry | Mapping[str, Any],
) -> np.ndarray:
    """Scale xy points from prediction space to canonical preprocessing space."""
    value = np.asarray(points)
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


def decollate_predict_full(output: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Split a validated full output on its normalized batch axis."""
    batch_size = int(np.asarray(output["aggregate"]).shape[0])

    def take(value: Any, index: int) -> Any:
        if (
            isinstance(value, np.ndarray)
            and value.ndim
            and value.shape[0] == batch_size
        ):
            return value[index]
        if isinstance(value, list) and len(value) == batch_size:
            return value[index]
        if isinstance(value, Mapping):
            return {key: take(item, index) for key, item in value.items()}
        return value

    return [
        {key: take(value, index) for key, value in output.items()}
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
