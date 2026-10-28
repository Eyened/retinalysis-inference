"""Inference input records and normalization at the dataset boundary."""

from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
from pydantic import AfterValidator, BeforeValidator, ConfigDict, TypeAdapter
from typing_extensions import TypedDict


def normalize_id(value):
    if value is None or isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, (str, Integral)):
        return str(value)
    raise TypeError(f"id must be a string or integer (got {type(value)!r})")


def validate_ids(identifiers, *, for_output=False, seen=None):
    """Normalize IDs and reject duplicates, optionally requiring file-safe IDs.

    Missing IDs are allowed for in-memory results and are never duplicates.
    ``seen`` tracks identifiers across batches within one inference run.
    """
    identifiers = [normalize_id(value) for value in identifiers]
    current = set()
    for identifier in identifiers:
        if for_output:
            if identifier is None:
                raise ValueError("Writing outputs requires an id for every input")
            if identifier in {".", ".."} or any(
                c in identifier for c in ("/", "\\", "\0")
            ):
                raise ValueError(
                    f"Output id must be a non-empty filename component: {identifier!r}"
                )
        if identifier is None:
            continue
        if identifier in current or seen is not None and identifier in seen:
            raise ValueError(f"Duplicate input id: {identifier!r}")
        current.add(identifier)
    if seen is not None:
        seen.update(current)
    return identifiers


def batch_ids(batch):
    """Read optional identifiers from a tensor batch, retaining input order."""
    count = int(batch["image"].shape[0])
    identifiers = batch.get("id")
    if identifiers is None:
        return [None] * count
    if isinstance(identifiers, torch.Tensor):
        identifiers = identifiers.detach().cpu().tolist()
    elif isinstance(identifiers, np.ndarray):
        identifiers = identifiers.tolist()
    if isinstance(identifiers, (str, Integral)):
        identifiers = [identifiers]
    if len(identifiers) != count:
        raise ValueError("ID and image batch sizes differ")
    return validate_ids(identifiers)


def loaded_rgb(image):
    """Validate RGB uint8 HWC arrays / CHW tensors; return CPU HWC pixels."""
    if isinstance(image, torch.Tensor):
        if image.dtype != torch.uint8:
            raise ValueError(
                f"Loaded image tensors must be RGB uint8, not normalized floats (got {image.dtype})"
            )
        if image.ndim != 3 or image.shape[0] != 3 or min(image.shape) == 0:
            raise ValueError(
                f"Loaded image tensors must have shape (3, H, W), got {tuple(image.shape)}"
            )
        image = image.detach().cpu().permute(1, 2, 0).numpy()
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            raise ValueError(
                f"Loaded image arrays must be RGB uint8, not normalized floats (got {image.dtype})"
            )
        if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape) == 0:
            raise ValueError(
                f"Loaded image arrays must have shape (H, W, 3), got {image.shape}"
            )
        return np.ascontiguousarray(image)
    return image


def normalize_path(value):
    return str(value) if isinstance(value, Path) else value


def normalize_fov(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"fov must be numeric, got {type(value).__name__}")
    return float(value)


def source_from_metadata(item):
    """Read the dataset source from a dictionary or a training DTO."""
    metadata = (
        item.get("metadata")
        if isinstance(item, Mapping)
        else getattr(item, "metadata", None)
    )
    source = (
        metadata.get("source")
        if isinstance(metadata, Mapping)
        else getattr(metadata, "source", None)
    )
    return (str(source).strip() or None) if source is not None else None


class ImageInput(TypedDict, total=False):
    """Input fields only; validation returns ordinary dictionaries.

    Primary images accept paths, RGB uint8 HWC arrays, or RGB uint8 CHW tensors.
    Auxiliary images and masks remain path-based. Metadata may have extra keys.
    """

    __pydantic_config__ = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    id: str | None
    image: str | Path | np.ndarray | torch.Tensor | None
    contrast_enhanced: str | Path | None
    input_mask: str | Path | None
    metadata: dict[str, Any] | None
    fov: float | None


def _normalize_input(value):
    if not isinstance(value, Mapping):
        raise ValueError("Each inference input must be a mapping")
    item = dict(value)
    if "image" in item:
        item["image"] = loaded_rgb(normalize_path(item["image"]))
    for name in ("contrast_enhanced", "input_mask"):
        if name in item:
            item[name] = normalize_path(item[name])
    identifier = normalize_id(item.get("id"))
    if identifier is None and isinstance(item.get("image"), str) and item["image"]:
        identifier = normalize_id(Path(item["image"]).stem)
    item["id"] = identifier
    if "fov" in item:
        item["fov"] = normalize_fov(item["fov"])
    metadata = item.get("metadata")
    if isinstance(metadata, Mapping):
        source = metadata.get("source")
        if source is not None and not isinstance(source, str):
            raise ValueError("metadata.source must be a string or None")
        item["metadata"] = {
            key: value for key, value in metadata.items() if value is not None
        }
    return item


def _omit_none(item):
    return {key: value for key, value in item.items() if value is not None}


# Build the schema once; validate collections before dataloader workers start.
image_inputs_adapter = TypeAdapter(
    list[
        Annotated[
            ImageInput, BeforeValidator(_normalize_input), AfterValidator(_omit_none)
        ]
    ]
)
