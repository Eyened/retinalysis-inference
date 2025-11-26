from __future__ import annotations

import numbers
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelInputDTO(BaseModel):
    """Validated representation of a single inference input."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Unique identifier for the sample.")
    image: Optional[str] = Field(
        default=None, description="Path to the primary RGB fundus image."
    )
    contrast_enhanced: Optional[str] = Field(
        default=None, description="Path to the contrast enhanced image (optional)."
    )
    metadata: Optional[Any] = Field(
        default=None,
        description="Arbitrary metadata that should be kept alongside the sample.",
    )
    fov: Optional[float] = Field(
        default=None, description="Optional field-of-view measurement."
    )

    @staticmethod
    def _validate_existing_path(path_value: str, field_name: str) -> str:
        if not isinstance(path_value, str):
            raise TypeError(f"{field_name} must be a string (got {type(path_value)!r})")
        resolved = Path(path_value).expanduser()
        if not resolved.exists():
            raise ValueError(f"{field_name} path does not exist: {path_value}")
        return str(resolved)

    @staticmethod
    def _ensure_float(value: Any, field_name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{field_name} cannot be boolean")
        if isinstance(value, numbers.Real):
            return float(value)
        raise TypeError(f"{field_name} must be numeric (got {type(value)!r})")

    @field_validator("image", "contrast_enhanced", mode="before")
    @classmethod
    def validate_paths(cls, value: Optional[str], info):
        if value in (None, ""):
            return None
        return cls._validate_existing_path(value, info.field_name)

    @field_validator("id", mode="before")
    @classmethod
    def validate_id(cls, value: Any):
        """Convert id to string, accepting both str and int."""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, numbers.Integral)):
            return str(value)
        raise TypeError(f"id must be a string or integer (got {type(value)!r})")

    @field_validator("fov", mode="before")
    @classmethod
    def validate_fov(cls, value: Any, info):
        if value is None:
            return None
        return cls._ensure_float(value, info.field_name)

    def to_serialized_dict(self) -> Dict[str, Any]:
        """Serialize to the legacy dict format expected by FundusDataset."""
        return self.model_dump(exclude_none=True)


class InferenceDatasetDTO(BaseModel):
    """Container for datasets passed into inference."""

    model_config = ConfigDict(extra="forbid")

    images: List[ModelInputDTO] = Field(
        ..., description="Validated list of inference inputs."
    )

    def to_serialized_images(self) -> List[Dict[str, Any]]:
        """Serialize all images to the list-of-dicts format."""
        return [image.to_serialized_dict() for image in self.images]
