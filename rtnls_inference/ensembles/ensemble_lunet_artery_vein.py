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
from rtnls_inference.ensembles.predict_output import restore_array_to_preprocessed


class LUNetArteryVeinEnsemble(SegmentationEnsembleOverlaps):
    """Halo-tile inference for LUNet's artery, vein, and vessel logits."""

    def _halo_logits(self, image: torch.Tensor) -> torch.Tensor:
        model_config = self.config.get("lightningmodule", {})
        return halo_sliding_window_inference(
            image,
            self.ensemble,
            context_size=int(model_config.get("context_size", 1024)),
            output_size=int(model_config.get("output_size", 512)),
            overlap=float(self._inference_setting("overlap", 0.5)),
            sw_batch_size=self._tile_batch_size(1, legacy_batch_size=True),
            sigma_scale=float(self._inference_setting("gaussian_sigma_scale", 0.125)),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return N,M,C,H,W logits, preserving the ensemble-model axis."""
        logits = self._halo_logits(image)
        flip_axes = self._inference_setting("tta_flips", [[2], [3], [2, 3]])
        if not self._inference_setting("tta", False):
            return logits
        for axes in flip_axes:
            flipped = self._halo_logits(torch.flip(image, dims=axes))
            logits += torch.flip(flipped, dims=[axis + 1 for axis in axes])
        return logits / (len(flip_axes) + 1)

    def postprocess_item(self, item):
        probabilities = restore_array_to_preprocessed(
            item["aggregate"], item["geometry"], "bilinear"
        )
        artery = probabilities[..., 0] > 0.5
        vein = probabilities[..., 1] > 0.5
        mask = artery.astype(np.uint8) + 2 * vein.astype(np.uint8)
        result = {
            "id": item.get("id"),
            "output": mask,
            "probabilities": probabilities,
            "output_kind": "artery_vein_mask",
            "output_space": "preprocessed",
            "geometry": item["geometry"],
        }
        if "preprocessed_image" in item:
            result["preprocessed_image"] = item["preprocessed_image"]
        return result

    def _save_item(self, item: dict, dest_path: str | Path):
        Image.fromarray(np.asarray(item["output"], dtype=np.uint8)).save(dest_path)
