"""Small classification metadata checks shared by inference and release export."""

from collections.abc import Mapping

import numpy as np


def validate_classification_config(
    metadata: Mapping, n_class: int | None = None
) -> dict:
    """Return a normalized copy, checking class semantics without a config schema.

    Preserve extra settings and leave defaults to the caller. JSON label-map
    keys become integers; the supplied configuration is never modified.
    """
    if not isinstance(metadata, Mapping):
        raise TypeError("classification metadata must be a mapping")
    config = dict(metadata)
    if config.get("task") not in {"multiclass", "multilabel"}:
        raise ValueError("classification task must be 'multiclass' or 'multilabel'")

    label_map = config.get("label_map")
    if label_map is not None:
        if not isinstance(label_map, Mapping):
            raise ValueError("label_map must map class indices to names")
        labels = {}
        for index, name in label_map.items():
            try:
                key = int(index)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("label_map indices must be integers") from error
            if not isinstance(index, str) and key != index:
                raise ValueError("label_map indices must be integers")
            if key in labels:
                raise ValueError("label_map indices must be unique")
            labels[key] = name
        if not labels or set(labels) != set(range(len(labels))):
            raise ValueError(
                "label_map must contain consecutive class indices starting at 0"
            )
        names = list(labels.values())
        if any(not isinstance(name, str) or not name.strip() for name in names) or len(
            set(names)
        ) != len(names):
            raise ValueError("label_map names must be non-empty and unique")
        config["label_map"] = labels

    if "threshold" in config:
        if config["task"] == "multiclass":
            raise ValueError(
                "threshold is only supported for multilabel classification"
            )
        threshold = np.asarray(config["threshold"], dtype=float)
        if (
            threshold.ndim > 1
            or threshold.size == 0
            or not np.isfinite(threshold).all()
            or ((threshold < 0) | (threshold > 1)).any()
        ):
            raise ValueError(
                "threshold must contain finite probabilities between 0 and 1"
            )
        config["threshold"] = threshold.tolist()

    if n_class is None and label_map is not None:
        n_class = len(config["label_map"])
    if n_class is not None:
        validate_classification_channels(config, n_class)
    return config


def validate_classification_channels(config: Mapping, n_class: int) -> None:
    """Check configured labels and thresholds against actual output channels."""
    if n_class < 1:
        raise ValueError("Classification requires at least one output channel")
    label_map = config.get("label_map")
    if label_map is not None and len(label_map) != n_class:
        raise ValueError(f"label_map must describe all {n_class} output channels")
    threshold = config.get("threshold", 0.5)
    if isinstance(threshold, list) and len(threshold) != n_class:
        raise ValueError(
            f"threshold must have one value per output channel ({n_class})"
        )
