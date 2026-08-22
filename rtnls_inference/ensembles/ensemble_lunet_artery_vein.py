from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rtnls_inference.ensembles.ensemble_artery_vein import (
    halo_sliding_window_inference,
)
from rtnls_inference.ensembles.ensemble_segmentation_overlaps import (
    SegmentationEnsembleOverlaps,
)


class LUNetArteryVeinEnsemble(SegmentationEnsembleOverlaps):
    """Halo-tile inference for LUNet's artery, vein, and vessel logits."""

    def _halo_logits(self, image: torch.Tensor) -> torch.Tensor:
        inference = self.config.get("inference", {})
        model_config = self.config.get("lightningmodule", {})
        return halo_sliding_window_inference(
            image,
            self.ensemble,
            context_size=int(model_config.get("context_size", 1024)),
            output_size=int(model_config.get("output_size", 512)),
            overlap=float(inference.get("overlap", 0.5)),
            sw_batch_size=int(inference.get("batch_size", 1)),
            sigma_scale=float(inference.get("gaussian_sigma_scale", 0.125)),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return N,M,C,H,W logits, preserving the ensemble-model axis."""
        logits = self._halo_logits(image)
        inference = self.config.get("inference", {})
        flip_axes = inference.get("tta_flips", [[2], [3], [2, 3]])
        if not inference.get("tta", False):
            return logits
        for axes in flip_axes:
            flipped = self._halo_logits(torch.flip(image, dims=axes))
            logits += torch.flip(flipped, dims=[axis + 1 for axis in axes])
        return logits / (len(flip_axes) + 1)

    def predict_step(self, batch, batch_idx=None):
        logits = self.forward(batch["image"]).mean(dim=1)
        return torch.sigmoid(logits).permute(0, 2, 3, 1)

    def _save_item(self, item: dict, dest_path: str | Path):
        probabilities = np.asarray(item["image"])
        artery = probabilities[..., 0] > 0.5
        vein = probabilities[..., 1] > 0.5
        # 0=background, 1=artery, 2=vein, 3=A/V overlap or crossing.
        mask = artery.astype(np.uint8) + 2 * vein.astype(np.uint8)
        Image.fromarray(mask).save(dest_path)
