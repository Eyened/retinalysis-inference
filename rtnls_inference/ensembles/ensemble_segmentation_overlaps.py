from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rtnls_inference.ensembles.ensemble_segmentation import (
    SegmentationEnsemble,
    SegmentationPredictFull,
)
from rtnls_inference.ensembles.predict_output import restore_array_to_preprocessed


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def flip(data, axis):
    return torch.flip(data, dims=axis)


class OverlapsPredictFull(SegmentationPredictFull):
    """Validated member logits and sigmoid aggregate for overlapping classes."""


class SegmentationEnsembleOverlaps(SegmentationEnsemble):
    """Ensemble for overlapping segmentation tasks.
    In contrast with SegmentationEnsemble which is post-processed with softmax over logits,
    this ensemble outputs n_class logit maps representing overlapping classes, which are processed with sigmoid
    """

    predict_full_model = OverlapsPredictFull

    def _aggregate_tensors(self, member_output):
        return torch.sigmoid(member_output.mean(dim=1))

    def _predict_step_tensors(self, batch):
        prediction = self._predict_member_tensors(batch)
        expected_channels = self.config.get("lightningmodule", {}).get("n_class")
        if expected_channels is not None and prediction.shape[-1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} segmentation channels, "
                f"got {prediction.shape[-1]}"
            )
        logits = prediction.mean(dim=1)
        aggregate = self._aggregate_tensors(prediction)
        return {"prediction": prediction, "logits": logits, "aggregate": aggregate}

    def postprocess_item(self, item):
        probabilities = restore_array_to_preprocessed(
            item["aggregate"], item["geometry"], "bilinear"
        )
        mask = probabilities > 0.5
        result = {
            "id": item.get("id"),
            "output": mask,
            "probabilities": probabilities,
            "output_kind": "multilabel_mask",
            "output_space": "preprocessed",
            "geometry": item["geometry"],
        }
        if "preprocessed_image" in item:
            result["preprocessed_image"] = item["preprocessed_image"]
        return result

    def _save_item(self, item: dict, dest_path: str | Path):
        r = item["output"][..., 0]
        b = item["output"][..., 1]
        im = np.stack([r, np.zeros_like(r), b], axis=-1)
        Image.fromarray(im.astype(np.uint8) * 255).save(dest_path)
