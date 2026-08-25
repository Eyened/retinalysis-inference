from __future__ import annotations

import torch

from rtnls_inference.ensembles.ensemble_artery_vein import (
    halo_sliding_window_inference,
)
from rtnls_inference.ensembles.ensemble_segmentation import SegmentationEnsemble


class HaloSegmentationEnsemble(SegmentationEnsemble):
    """Generic softmax segmentation ensemble for center-cropped halo models."""

    def sliding_window_inference(self, image: torch.Tensor) -> torch.Tensor:
        inference = self.config.get("inference", {})
        model_config = self.config.get("lightningmodule", {})
        return halo_sliding_window_inference(
            image,
            self.ensemble,
            context_size=int(model_config.get("context_size", 768)),
            output_size=int(model_config.get("output_size", 512)),
            overlap=float(inference.get("overlap", 0.5)),
            sw_batch_size=int(inference.get("batch_size", 1)),
            sigma_scale=float(inference.get("gaussian_sigma_scale", 0.125)),
        )
