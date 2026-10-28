import numpy as np


from .classification_output import (
    validate_classification_channels,
    validate_classification_config,
)
from .ensemble_regression import RegressionEnsemble, softmax


class ClassificationEnsemble(RegressionEnsemble):
    prediction_prefix = "logit"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        metadata = self._inference_config().get("classification")
        # Older releases without metadata retain their numeric-only results.
        self._classification_config = (
            validate_classification_config(
                metadata, n_class=self.config.get("lightningmodule", {}).get("n_class")
            )
            if metadata is not None
            else None
        )

    def _class_names(self, count):
        label_map = (self._classification_config or {}).get("label_map")
        return [label_map[i] if label_map is not None else str(i) for i in range(count)]

    def postprocess_item(self, item):
        result = super().postprocess_item(item)
        config = self._classification_config
        if config is None:
            return result

        logits = result["output"]
        if logits.ndim != 1 or not np.isfinite(logits).all():
            raise ValueError(
                "Classification logits must be a finite one-dimensional array"
            )
        validate_classification_channels(config, len(logits))
        label_map = config.get("label_map")
        # Keep output unchanged, including its dtype; compute probabilities with
        # at least float32 precision after averaging member logits.
        scores = logits.astype(np.result_type(logits.dtype, np.float32), copy=False)
        if config["task"] == "multiclass":
            class_id = int(scores.argmax())
            result.update(
                probabilities=softmax(scores),
                class_id=class_id,
                label=label_map[class_id] if label_map is not None else None,
            )
        else:
            probabilities = np.exp(-np.logaddexp(0, -scores))
            class_ids = np.flatnonzero(
                probabilities > np.asarray(config.get("threshold", 0.5))
            ).tolist()
            result.update(
                probabilities=probabilities,
                class_ids=class_ids,
                labels=[label_map[i] for i in class_ids]
                if label_map is not None
                else None,
            )
        return result

    def _prediction_columns(self, count):
        return [f"logit_{name}" for name in self._class_names(count)]

    def _summary_columns(self, count=0):
        config = self._classification_config
        count = count or self.config.get("lightningmodule", {}).get("n_class", 0)
        if config is None:
            return ["id", *self._prediction_columns(count)]
        count = count or len(config.get("label_map") or {})
        names = self._class_names(count) if count else []
        selected = (
            ["class_id", "label"]
            if config["task"] == "multiclass"
            else ["class_ids", "labels"]
        )
        return [
            "id",
            *selected,
            *[f"probability_{name}" for name in names],
            *self._prediction_columns(count),
        ]

    def _summary_row(self, item):
        row = super()._summary_row(item)
        config = self._classification_config
        if config is None:
            return row
        names = self._class_names(len(item["output"]))
        selected = (
            ["class_id", "label"]
            if config["task"] == "multiclass"
            else ["class_ids", "labels"]
        )
        return {
            "id": row.pop("id"),
            **{name: item[name] for name in selected},
            **{
                f"probability_{name}": value
                for name, value in zip(names, item["probabilities"])
            },
            **row,
        }
