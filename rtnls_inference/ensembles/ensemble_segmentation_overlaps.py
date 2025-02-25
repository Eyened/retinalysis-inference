from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rtnls_inference.ensembles.ensemble_segmentation import SegmentationEnsemble


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def flip(data, axis):
    return torch.flip(data, dims=axis)


class SegmentationEnsembleOverlaps(SegmentationEnsemble):
    """Ensemble for overlapping segmentation tasks.
    In contrast with SegmentationEnsemble which is post-processed with softmax over logits,
    this ensemble outputs n_class logit maps representing overlapping classes, which are processed with sigmoid
    """

    def predict_step(self, batch, batch_idx=None):
        """Returns the output averaged over models, shape NHWC"""
        logits = self.forward(batch["image"])
        logits = torch.mean(logits, dim=0)  # average logits over models
        logits = torch.permute(logits, (0, 2, 3, 1))  # NCHW -> NHWC
        proba = torch.sigmoid(logits)
        return proba

    def _save_item(self, item: dict, dest_path: str | Path):
        r = item["image"][..., 0] > 0.5
        b = item["image"][..., 1] > 0.5
        im = np.stack([r, np.zeros_like(r), b], axis=-1)
        Image.fromarray(im.astype(np.uint8) * 255).save(dest_path)
