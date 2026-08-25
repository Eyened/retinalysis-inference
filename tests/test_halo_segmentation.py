from __future__ import annotations

import numpy as np
import torch
from rtnls_inference.ensembles import (
    HaloSegmentationEnsemble,
    get_ensemble_class,
)


class ConstantHaloMembers(torch.nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size
        self.tile_count = 0

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.tile_count += image.shape[0]
        foreground = torch.ones(
            (image.shape[0], self.output_size, self.output_size),
            device=image.device,
            dtype=image.dtype,
        )
        logits = torch.stack([torch.zeros_like(foreground), foreground], dim=1)
        return logits[None]


def make_ensemble(*, tta: bool = False):
    members = ConstantHaloMembers(output_size=512)
    config = {
        "lightningmodule": {
            "n_class": 2,
            "context_size": 768,
            "output_size": 512,
        },
        "inference": {
            "ensemble_class": "HaloSegmentationEnsemble",
            "overlap": 0.5,
            "batch_size": 16,
            "tta": tta,
            "tta_flips": [[2], [3], [2, 3]],
        },
    }
    return HaloSegmentationEnsemble(members, config), members


def test_halo_ensemble_reconstructs_1280_probabilities_and_binary_mask():
    ensemble, members = make_ensemble()
    batch = {"id": ["drusen"], "image": torch.zeros((1, 3, 1280, 1280))}

    full = ensemble.predict_step_full(batch)

    assert full["prediction"].shape == (1, 1, 1280, 1280, 2)
    assert full["logits"].shape == (1, 1280, 1280, 2)
    assert full["aggregate"].shape == (1, 1280, 1280, 2)
    np.testing.assert_allclose(full["aggregate"].sum(axis=-1), 1.0, atol=1e-6)
    assert members.tile_count == 16

    item = ensemble._predict_output_batch(batch)[0]
    assert item["output"].shape == (1280, 1280)
    assert item["output"].dtype == np.uint8
    assert np.all(item["output"] == 1)


def test_halo_ensemble_tta_keeps_shape_and_undoes_flips():
    ensemble, members = make_ensemble(tta=True)
    output = ensemble.forward(torch.zeros((1, 3, 1280, 1280)))

    assert output.shape == (1, 1, 2, 1280, 1280)
    assert members.tile_count == 64
    torch.testing.assert_close(output[:, :, 0], torch.zeros_like(output[:, :, 0]))
    torch.testing.assert_close(output[:, :, 1], torch.ones_like(output[:, :, 1]))


def test_halo_ensemble_is_discoverable_from_release_config():
    config = {"inference": {"ensemble_class": "HaloSegmentationEnsemble"}}
    assert get_ensemble_class(config) is HaloSegmentationEnsemble
