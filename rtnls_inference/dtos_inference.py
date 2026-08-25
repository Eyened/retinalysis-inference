from __future__ import annotations

import numbers
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SampleMetadataDTO(BaseModel):
    """Per-sample metadata passed through dataloaders.

    Extra keys such as `bounds` are allowed. `source` names the originating
    dataset when present and is used to stratify release evaluation.
    """

    model_config = ConfigDict(extra="allow")

    source: Optional[str] = Field(
        default=None,
        description="Optional dataset or collection name used to stratify evaluation.",
    )


def source_from_metadata(item: Any) -> Optional[str]:
    """Return metadata.source from a DTO, dumped dict, or collated batch item."""
    metadata = None
    if isinstance(item, dict):
        metadata = item.get("metadata")
    else:
        metadata = getattr(item, "metadata", None)
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        source = metadata.get("source")
    else:
        source = getattr(metadata, "source", None)
    if source is None:
        return None
    source = str(source).strip()
    return source or None


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
    input_mask: Optional[str] = Field(
        default=None,
        description="Path to an optional input segmentation mask (binary).",
    )
    metadata: Optional[SampleMetadataDTO] = Field(
        default=None,
        description="Optional sample metadata. Set metadata.source to name the originating dataset.",
    )
    fov: Optional[float] = Field(
        default=None, description="Optional field-of-view measurement."
    )



    @staticmethod
    def _ensure_float(value: Any, field_name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{field_name} cannot be boolean")
        if isinstance(value, numbers.Real):
            return float(value)
        raise TypeError(f"{field_name} must be numeric (got {type(value)!r})")



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
