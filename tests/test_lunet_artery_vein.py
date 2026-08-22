import torch
from rtnls_inference.ensembles import LUNetArteryVeinEnsemble


class _CentralLUNetEnsemble(torch.nn.Module):
    def forward(self, image):
        central = image[:, :1, 16:48, 16:48]
        artery_vein = torch.cat([central, -central], dim=1)
        vessel = artery_vein.amax(dim=1, keepdim=True)
        # Model releases retain an explicit ensemble-model axis.
        return torch.cat([artery_vein, vessel], dim=1)[None]


def test_lunet_ensemble_stitches_halo_logits_and_returns_overlap_probabilities():
    config = {
        "lightningmodule": {"context_size": 64, "output_size": 32},
        "inference": {"overlap": 0.5, "batch_size": 1, "tta": False},
    }
    ensemble = LUNetArteryVeinEnsemble(_CentralLUNetEnsemble(), config)
    batch = {"image": torch.ones((1, 3, 64, 64))}

    logits = ensemble.forward(batch["image"])
    probabilities = ensemble.predict_step(batch)

    assert logits.shape == (1, 1, 3, 64, 64)
    assert probabilities.shape == (1, 64, 64, 3)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    torch.testing.assert_close(
        probabilities[..., 2],
        torch.maximum(probabilities[..., 0], probabilities[..., 1]),
    )
