from __future__ import annotations

import numpy as np
import pytest
import torch
from rtnls_inference.artery_vein import (
    AV_HEAD_NAMES,
    av_logits_to_legacy_probabilities,
    load_av_head_logits,
    save_av_head_logits,
)
from rtnls_inference.ensembles.ensemble_artery_vein import (
    AV_REFINEMENT_VARIANTS,
    artery_vein_refinement_variants,
    halo_sliding_window_inference,
    normalize_refinement_mode,
    refine_artery_vein,
    refine_artery_vein_graph,
)


class _CenterPredictor(torch.nn.Module):
    def forward(self, image):
        center = image[:, :1, 8:24, 8:24]
        return center.repeat(1, len(AV_HEAD_NAMES), 1, 1)


class _RecordingCenterPredictor(_CenterPredictor):
    def __init__(self):
        super().__init__()
        self.tiles = []

    def forward(self, image):
        self.tiles.append(image.detach().clone())
        return super().forward(image)


class _CenterEnsemble(torch.nn.Module):
    def forward(self, image):
        prediction = _CenterPredictor()(image)
        return torch.stack([prediction, prediction + 2.0])


def test_named_head_logits_roundtrip(tmp_path):
    logits = np.random.default_rng(3).normal(size=(23, 29, 4)).astype(np.float32)
    path = tmp_path / "sample.npz"
    save_av_head_logits(path, logits)
    loaded = load_av_head_logits(path)
    assert loaded.shape == logits.shape
    np.testing.assert_allclose(loaded, logits, atol=2e-3)


def test_named_head_logits_rejects_legacy_tensor(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, logits=np.zeros((20, 20, 4), dtype=np.float16))
    try:
        load_av_head_logits(path)
    except ValueError as error:
        assert "Invalid AV head-logit artifact" in str(error)
    else:
        raise AssertionError("Legacy logits should not be accepted")


def test_named_head_logits_rejects_schema_v1(tmp_path):
    path = tmp_path / "schema_v1.npz"
    payload = {name: np.zeros((20, 20), dtype=np.float16) for name in AV_HEAD_NAMES}
    np.savez(
        path,
        schema_version=np.asarray(1, dtype=np.int16),
        source_shape=np.asarray((20, 20), dtype=np.int32),
        **payload,
    )
    try:
        load_av_head_logits(path)
    except ValueError as error:
        assert "Unsupported AV head-logit schema 1" in str(error)
    else:
        raise AssertionError("Schema-v1 logits should not be accepted")


def test_named_head_logits_rejects_malformed_shape_and_nonfinite_values(tmp_path):
    malformed_path = tmp_path / "malformed.npz"
    payload = {name: np.zeros((20, 20), dtype=np.float16) for name in AV_HEAD_NAMES}
    payload["artery"] = np.zeros((19, 20), dtype=np.float16)
    np.savez(
        malformed_path,
        schema_version=np.asarray(2, dtype=np.int16),
        source_shape=np.asarray((20, 20), dtype=np.int32),
        **payload,
    )
    with pytest.raises(ValueError, match="has shape"):
        load_av_head_logits(malformed_path)

    nonfinite = np.zeros((20, 20, 4), dtype=np.float32)
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        save_av_head_logits(tmp_path / "nonfinite.npz", nonfinite)


def test_halo_stitching_reconstructs_center_predictions():
    image = torch.linspace(-1, 1, 70 * 74).reshape(1, 1, 70, 74)
    output = halo_sliding_window_inference(
        image,
        _CenterPredictor(),
        context_size=32,
        output_size=16,
        overlap=0.5,
        sw_batch_size=3,
    )
    assert output.shape == (1, 4, 70, 74)
    torch.testing.assert_close(output[:, 0], image[:, 0], atol=1e-5, rtol=1e-5)


def test_halo_stitching_uses_zero_context_padding():
    predictor = _RecordingCenterPredictor()
    image = torch.ones((1, 1, 16, 16))

    output = halo_sliding_window_inference(
        image,
        predictor,
        context_size=32,
        output_size=16,
        overlap=0.5,
    )

    tile = predictor.tiles[0]
    assert torch.count_nonzero(tile[..., :8, :]) == 0
    assert torch.count_nonzero(tile[..., -8:, :]) == 0
    assert torch.count_nonzero(tile[..., :, :8]) == 0
    assert torch.count_nonzero(tile[..., :, -8:]) == 0
    torch.testing.assert_close(tile[..., 8:24, 8:24], image)
    torch.testing.assert_close(output[:, 0], image[:, 0])


def test_halo_stitching_preserves_model_axis():
    image = torch.zeros((2, 1, 40, 42))
    output = halo_sliding_window_inference(
        image,
        _CenterEnsemble(),
        context_size=32,
        output_size=16,
        overlap=0.5,
        sw_batch_size=2,
    )
    assert output.shape == (2, 2, 4, 40, 42)
    torch.testing.assert_close(
        output[:, 1] - output[:, 0], torch.full_like(output[:, 0], 2)
    )


def test_legacy_projection_is_normalized():
    logits = np.random.default_rng(5).normal(size=(17, 19, 4)).astype(np.float32)
    probabilities = av_logits_to_legacy_probabilities(logits)
    assert probabilities.shape == (17, 19, 4)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-6)


def test_graph_refinement_preserves_support_and_crossings():
    logits = np.full((48, 48, 4), -8.0, dtype=np.float32)
    logits[23:26, 5:43, 0] = 8.0
    logits[5:43, 23:26, 0] = 8.0
    logits[23:26, 5:43, 1] = 3.0
    logits[23:26, 5:43, 2] = -3.0
    logits[5:43, 23:26, 1] = -3.0
    logits[5:43, 23:26, 2] = 3.0
    logits[22:27, 22:27, 3] = 8.0
    refined = refine_artery_vein_graph(logits, min_component_size=3)
    vessel = logits[..., 0] > 0
    assert np.array_equal(refined > 0, vessel)
    assert np.all(refined[22:27, 22:27][vessel[22:27, 22:27]] == 3)


def test_refinement_mode_aliases():
    assert normalize_refinement_mode("none") == "basic"
    assert normalize_refinement_mode("simple") == "refinement_simple"
    assert normalize_refinement_mode("full") == "refinement_full"


def test_all_refinement_variants_preserve_vessels_and_crossings():
    logits = np.full((48, 48, 4), -8.0, dtype=np.float32)
    logits[23:26, 5:43, 0] = 8.0
    logits[5:43, 23:26, 0] = 8.0
    logits[23:26, 5:43, 1] = 3.0
    logits[23:26, 5:43, 2] = -3.0
    logits[5:43, 23:26, 1] = -3.0
    logits[5:43, 23:26, 2] = 3.0
    logits[22:27, 22:27, 3] = 8.0

    variants = artery_vein_refinement_variants(logits, min_component_size=3)

    assert tuple(variants) == AV_REFINEMENT_VARIANTS
    vessel = logits[..., 0] > 0
    crossing = vessel & (logits[..., 3] > 0)
    for result in variants.values():
        assert np.array_equal(result > 0, vessel)
        assert np.all(result[crossing] == 3)


def test_simple_scores_segments_without_directional_reconnection():
    logits = np.full((64, 64, 4), -8.0, dtype=np.float32)
    logits[30:33, 5:59, 0] = 8.0
    logits[5:59, 30:33, 0] = 8.0
    # The weak right-hand segment disagrees with the strong opposite segment.
    logits[30:33, 5:29, 1] = 6.0
    logits[30:33, 5:29, 2] = -6.0
    logits[30:33, 35:59, 1] = -1.0
    logits[30:33, 35:59, 2] = 1.0
    logits[5:29, 30:33, 1] = -6.0
    logits[5:29, 30:33, 2] = 6.0
    logits[35:59, 30:33, 1] = -6.0
    logits[35:59, 30:33, 2] = 6.0
    logits[28:35, 28:35, 3] = 8.0

    simple = refine_artery_vein(
        logits,
        "simple",
        crossing_radius=2,
        node_radius=1,
        min_component_size=3,
    )
    full = refine_artery_vein(
        logits,
        "full",
        crossing_radius=2,
        node_radius=1,
        min_component_size=3,
    )

    assert simple[31, 50] == 2
    assert full[31, 50] == 1
