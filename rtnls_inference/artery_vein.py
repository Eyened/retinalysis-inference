from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

AV_HEAD_LOGITS_SCHEMA_VERSION = 1
AV_HEAD_NAMES = (
    "vesselness",
    "artery",
    "vein",
    "crossing",
    "centerline_vessel",
    "centerline_artery",
    "centerline_vein",
)


def load_av_head_logits(path: str | Path) -> np.ndarray:
    """Load a validated AV head-logit artifact as an HWC float32 array."""
    with np.load(path, allow_pickle=False) as artifact:
        keys = set(artifact.files)
        expected = {*AV_HEAD_NAMES, "schema_version", "source_shape"}
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            raise ValueError(
                f"Invalid AV head-logit artifact {path}: missing={missing}, extra={extra}"
            )

        version = int(np.asarray(artifact["schema_version"]).reshape(-1)[0])
        if version != AV_HEAD_LOGITS_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported AV head-logit schema {version}; "
                f"expected {AV_HEAD_LOGITS_SCHEMA_VERSION}"
            )

        source_shape = tuple(
            int(v) for v in np.asarray(artifact["source_shape"]).tolist()
        )
        if len(source_shape) != 2 or min(source_shape) <= 0:
            raise ValueError(f"Invalid source_shape in {path}: {source_shape}")

        heads = []
        for name in AV_HEAD_NAMES:
            value = np.asarray(artifact[name])
            if value.ndim != 2 or value.shape != source_shape:
                raise ValueError(
                    f"Head {name!r} in {path} has shape {value.shape}; "
                    f"expected {source_shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"Head {name!r} in {path} contains non-finite values")
            heads.append(value.astype(np.float32, copy=False))

    return np.stack(heads, axis=-1)


def save_av_head_logits(
    path: str | Path,
    logits: np.ndarray | Mapping[str, np.ndarray],
    dtype: np.dtype | type = np.float16,
) -> None:
    """Save AV pre-sigmoid logits using the versioned named NPZ schema."""
    if isinstance(logits, Mapping):
        missing = [name for name in AV_HEAD_NAMES if name not in logits]
        extra = [name for name in logits if name not in AV_HEAD_NAMES]
        if missing or extra:
            raise ValueError(f"Invalid AV heads: missing={missing}, extra={extra}")
        arrays = {name: np.asarray(logits[name]) for name in AV_HEAD_NAMES}
    else:
        value = np.asarray(logits)
        if value.ndim != 3 or value.shape[-1] != len(AV_HEAD_NAMES):
            raise ValueError(
                f"AV logits must have HWC shape with seven channels, got {value.shape}"
            )
        arrays = {name: value[..., idx] for idx, name in enumerate(AV_HEAD_NAMES)}

    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"AV head shapes do not match: {sorted(shapes)}")
    source_shape = next(iter(shapes))
    if len(source_shape) != 2 or min(source_shape) <= 0:
        raise ValueError(f"Invalid AV source shape: {source_shape}")
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Head {name!r} contains non-finite values")

    payload = {name: value.astype(dtype, copy=False) for name, value in arrays.items()}
    np.savez_compressed(
        path,
        schema_version=np.asarray(AV_HEAD_LOGITS_SCHEMA_VERSION, dtype=np.int16),
        source_shape=np.asarray(source_shape, dtype=np.int32),
        **payload,
    )


def av_logits_to_legacy_probabilities(logits: np.ndarray) -> np.ndarray:
    """Project vessel, conditional A/V, and crossing logits to four classes."""
    logits = np.asarray(logits)
    if logits.shape[-1] != len(AV_HEAD_NAMES):
        raise ValueError(f"Expected seven AV logits, got shape {logits.shape}")
    vessel = 1.0 / (1.0 + np.exp(-np.clip(logits[..., 0], -80.0, 80.0)))
    crossing = 1.0 / (1.0 + np.exp(-np.clip(logits[..., 3], -80.0, 80.0)))
    av_logits = logits[..., 1:3]
    av_logits = av_logits - av_logits.max(axis=-1, keepdims=True)
    av_probability = np.exp(av_logits)
    av_probability /= av_probability.sum(axis=-1, keepdims=True)
    background = 1.0 - vessel
    crossing_class = vessel * crossing
    non_crossing = vessel * (1.0 - crossing)
    artery_class = non_crossing * av_probability[..., 0]
    vein_class = non_crossing * av_probability[..., 1]
    return np.stack(
        [background, artery_class, vein_class, crossing_class], axis=-1
    ).astype(np.float32, copy=False)
