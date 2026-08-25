from __future__ import annotations

import numpy as np
import pytest
import rtnls_inference.ensembles.ensemble_artery_vein as av_module
import torch
from pydantic import ValidationError
from rtnls_inference.artery_vein import AV_HEAD_NAMES
from rtnls_inference.ensembles.ensemble_artery_vein import (
    ArteryVeinPredictFull,
    ArteryVeinSegmentationEnsemble,
)
from rtnls_inference.ensembles.ensemble_classification import ClassificationEnsemble
from rtnls_inference.ensembles.ensemble_embedding import EmbeddingEnsemble
from rtnls_inference.ensembles.ensemble_heatmap_regression import (
    HeatmapRegressionEnsemble,
)
from rtnls_inference.ensembles.ensemble_keypoints import KeypointsEnsemble
from rtnls_inference.ensembles.ensemble_lunet_artery_vein import (
    LUNetArteryVeinEnsemble,
)
from rtnls_inference.ensembles.ensemble_regression import RegressionEnsemble
from rtnls_inference.ensembles.ensemble_segmentation import (
    SegmentationEnsemble,
    SegmentationPredictFull,
)
from rtnls_inference.ensembles.ensemble_segmentation_overlaps import (
    SegmentationEnsembleOverlaps,
)
from rtnls_inference.ensembles.predict_output import (
    PredictionGeometry,
    decollate_predict_full,
    restore_array_to_preprocessed,
    restore_points_to_preprocessed,
)

N = 2
M = 3
H = 4
W = 6


class _UnusedBackend(torch.nn.Module):
    def forward(self, image):  # pragma: no cover - direct test ensembles override it
        raise AssertionError("backend should not be called")


class _DirectSegmentation(SegmentationEnsemble):
    def forward(self, image):
        values = torch.arange(
            N * M * 3 * H * W, device=image.device, dtype=torch.float32
        )
        return values.reshape(N, M, 3, H, W) / 100.0


class _DirectOverlaps(SegmentationEnsembleOverlaps):
    def forward(self, image):
        values = torch.arange(
            N * M * 2 * H * W, device=image.device, dtype=torch.float32
        )
        return values.reshape(N, M, 2, H, W) / 100.0 - 1.0


class _DirectLUNet(LUNetArteryVeinEnsemble):
    def forward(self, image):
        values = torch.arange(
            N * M * 3 * H * W, device=image.device, dtype=torch.float32
        )
        return values.reshape(N, M, 3, H, W) / 100.0 - 1.0


class _MemberFirstBackend(torch.nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.features = features

    def forward(self, image):
        values = torch.arange(
            M * len(image) * self.features,
            device=image.device,
            dtype=torch.float32,
        )
        return values.reshape(M, len(image), self.features)


class _ScalarRegressionBackend(torch.nn.Module):
    def forward(self, image):
        return torch.arange(M * len(image), dtype=torch.float32).reshape(M, len(image))


class _DirectHeatmap(HeatmapRegressionEnsemble):
    def forward(self, image):
        heatmaps = torch.zeros((N, M, 2, H, W), device=image.device)
        for member in range(M):
            heatmaps[:, member, 0, 1, member + 1] = 10
            heatmaps[:, member, 1, 2, member + 2] = 10
        return heatmaps


class _DirectAV(ArteryVeinSegmentationEnsemble):
    def forward(self, image):
        values = torch.arange(
            N * M * 4 * H * W, device=image.device, dtype=torch.float32
        )
        return values.reshape(N, M, 4, H, W) / 100.0 - 2.0


def _config(**inference):
    return {
        "datamodule": {"normalize_keypoints": True},
        "lightningmodule": {},
        "inference": inference,
    }


def _batch(with_context: bool = True):
    batch = {
        "id": ["first", "second"],
        "image": torch.zeros((N, 3, H, W)),
    }
    if with_context:
        batch["preprocessed_image"] = torch.zeros(
            (N, H * 2, W * 2, 3), dtype=torch.uint8
        )
    return batch


@pytest.mark.parametrize(
    ("ensemble", "member_shape", "aggregate_shape"),
    [
        (
            _DirectSegmentation(_UnusedBackend(), _config()),
            (N, M, H, W, 3),
            (N, H, W, 3),
        ),
        (_DirectOverlaps(_UnusedBackend(), _config()), (N, M, H, W, 2), (N, H, W, 2)),
        (_DirectLUNet(_UnusedBackend(), _config()), (N, M, H, W, 3), (N, H, W, 3)),
        (RegressionEnsemble(_MemberFirstBackend(4), _config()), (N, M, 4), (N, 4)),
        (ClassificationEnsemble(_MemberFirstBackend(3), _config()), (N, M, 3), (N, 3)),
        (EmbeddingEnsemble(_MemberFirstBackend(5), _config()), (N, M, 5), (N, 5)),
        (KeypointsEnsemble(_MemberFirstBackend(4), _config()), (N, M, 2, 2), (N, 2, 2)),
        (_DirectHeatmap(_UnusedBackend(), _config()), (N, M, 2, 2), (N, 2, 2)),
        (
            _DirectAV(_UnusedBackend(), _config()),
            (N, M, H, W, 4),
            (N, H, W, 4),
        ),
    ],
)
def test_family_contracts_are_batch_major_and_aggregated(
    ensemble, member_shape, aggregate_shape
):
    batch = _batch()
    aggregate = ensemble.predict_step(batch)
    full = ensemble.predict_step_full(batch)

    assert aggregate.shape == aggregate_shape
    assert aggregate.device == batch["image"].device
    assert not aggregate.requires_grad
    assert full["prediction"].shape == member_shape
    assert full["aggregate"].shape == aggregate_shape
    assert isinstance(full, dict)
    assert isinstance(full["prediction"], np.ndarray)
    assert isinstance(full["aggregate"], np.ndarray)
    assert full["preprocessed_image"].shape == (N, H * 2, W * 2, 3)
    np.testing.assert_allclose(full["aggregate"], aggregate.numpy(), rtol=1e-5)


def test_spatial_and_av_intermediates_are_validated():
    segmentation = _DirectSegmentation(_UnusedBackend(), _config())
    segmentation_full = segmentation.predict_step_full(_batch())
    assert segmentation_full["logits"].shape == (N, H, W, 3)

    heatmap = _DirectHeatmap(_UnusedBackend(), _config()).predict_step_full(_batch())
    assert heatmap["heatmaps"].shape == (N, M, 2, H, W)

    av = _DirectAV(_UnusedBackend(), _config()).predict_step_full(_batch())
    assert av["logits"].shape == (N, H, W, 4)
    assert av["logit_names"] == tuple(AV_HEAD_NAMES)
    np.testing.assert_allclose(av["aggregate"].sum(axis=-1), 1.0, atol=1e-6)


def test_scalar_regression_is_promoted_to_nc_and_nmc():
    ensemble = RegressionEnsemble(_ScalarRegressionBackend(), _config())
    aggregate = ensemble.predict_step(_batch())
    full = ensemble.predict_step_full(_batch())
    assert aggregate.shape == (N, 1)
    assert full["prediction"].shape == (N, M, 1)


def test_direct_tensor_batch_uses_model_input_as_canonical_fallback():
    ensemble = _DirectSegmentation(_UnusedBackend(), _config())
    full = ensemble.predict_step_full(_batch(with_context=False))
    assert "preprocessed_image" not in full
    assert (
        full["geometry"]
        == [{"preprocessed_size": (H, W), "prediction_size": (H, W)}] * N
    )


def test_invalid_schema_shapes_and_batch_lengths_fail():
    with pytest.raises(ValidationError):
        SegmentationPredictFull.model_validate(
            {
                "prediction": np.zeros((N, M, H, W)),
                "aggregate": np.zeros((N, H, W, 2)),
                "logits": np.zeros((N, H, W, 2)),
            }
        )
    with pytest.raises(ValidationError):
        ArteryVeinPredictFull.model_validate(
            {
                "prediction": np.zeros((N, M, H, W, 6)),
                "aggregate": np.zeros((N, H, W, 4)),
                "logits": np.zeros((N, H, W, 4)),
                "logit_names": AV_HEAD_NAMES,
                "refinement_mode": "basic",
                "refinement_parameters": {},
            }
        )
    with pytest.raises(ValidationError):
        SegmentationPredictFull.model_validate(
            {
                "prediction": np.zeros((N, M, H, W, 2)),
                "aggregate": np.zeros((N, H, W, 2)),
                "logits": np.zeros((N, H, W, 2)),
                "geometry": [{"preprocessed_size": (H, W)}],
            }
        )


def test_model_dump_and_decollation_preserve_ndarrays():
    full = _DirectSegmentation(_UnusedBackend(), _config()).predict_step_full(_batch())
    items = decollate_predict_full(full)
    assert len(items) == N
    assert isinstance(items[0]["prediction"], np.ndarray)
    assert items[0]["prediction"].shape == (M, H, W, 3)
    assert items[1]["id"] == "second"


def test_resize_only_restoration_and_non_square_point_scaling():
    geometry = PredictionGeometry(preprocessed_size=(8, 12), prediction_size=(4, 3))
    continuous = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    half_precision = continuous.astype(np.float16)
    labels = (continuous > 5).astype(np.uint8)
    assert restore_array_to_preprocessed(continuous, geometry, "bilinear").shape == (
        8,
        12,
    )
    nearest = restore_array_to_preprocessed(labels, geometry, "nearest")
    assert nearest.dtype == np.uint8
    assert set(np.unique(nearest)) <= {0, 1}
    restored_half = restore_array_to_preprocessed(half_precision, geometry, "bilinear")
    assert restored_half.dtype == np.float16
    points = restore_points_to_preprocessed(np.array([[1.0, 2.0]]), geometry)
    np.testing.assert_allclose(points, [[4.0, 4.0]])

    canonical = restore_array_to_preprocessed(
        np.zeros((512, 512), dtype=np.float32),
        {"preprocessed_size": (1024, 1024), "prediction_size": (512, 512)},
        "bilinear",
    )
    assert canonical.shape == (1024, 1024)


def test_segmentation_restores_probabilities_before_argmax():
    ensemble = _DirectSegmentation(_UnusedBackend(), _config())
    probabilities = np.array(
        [
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.1, 0.9], [0.9, 0.1]],
        ],
        dtype=np.float32,
    )
    geometry = {"preprocessed_size": (5, 7), "prediction_size": (2, 2)}
    item = {"id": "x", "aggregate": probabilities, "geometry": geometry}
    processed = ensemble.postprocess_item(item)
    expected = np.argmax(
        restore_array_to_preprocessed(probabilities, geometry, "bilinear"), axis=-1
    )
    np.testing.assert_array_equal(processed["output"], expected)
    assert processed["output_space"] == "preprocessed"


@pytest.mark.parametrize(
    ("ensemble", "expected"),
    [
        (
            _DirectOverlaps(_UnusedBackend(), _config()),
            lambda probabilities: probabilities > 0.5,
        ),
        (
            _DirectLUNet(_UnusedBackend(), _config()),
            lambda probabilities: (
                (probabilities[..., 0] > 0.5).astype(np.uint8)
                + 2 * (probabilities[..., 1] > 0.5).astype(np.uint8)
            ),
        ),
    ],
)
def test_overlap_families_restore_probabilities_before_thresholding(ensemble, expected):
    probabilities = np.array(
        [
            [[0.25, 0.75, 0.0], [0.75, 0.25, 1.0]],
            [[0.75, 0.25, 1.0], [0.25, 0.75, 0.0]],
        ],
        dtype=np.float32,
    )
    if isinstance(ensemble, _DirectOverlaps):
        probabilities = probabilities[..., :2]
    geometry = {"preprocessed_size": (5, 7), "prediction_size": (2, 2)}
    item = {"id": "x", "aggregate": probabilities, "geometry": geometry}
    processed = ensemble.postprocess_item(item)
    restored = restore_array_to_preprocessed(probabilities, geometry, "bilinear")
    np.testing.assert_array_equal(processed["output"], expected(restored))
    assert processed["probabilities"].shape[:2] == (5, 7)


def test_halo_av_refines_before_nearest_restoration(monkeypatch):
    ensemble = _DirectAV(
        _UnusedBackend(),
        _config(graph_refinement={"mode": "full", "relabel_margin": 999.0}),
        refinement_mode="simple",
        refinement_parameters={"relabel_margin": 0.25},
    )
    full_item = decollate_predict_full(ensemble.predict_step_full(_batch()))[0]
    observed = []

    def fake_refine(logits, mode, **kwargs):
        observed.append((logits.shape, mode, kwargs))
        return np.ones(logits.shape[:2], dtype=np.uint8)

    monkeypatch.setattr(av_module, "refine_artery_vein", fake_refine)
    processed = ensemble.postprocess_item(full_item)
    assert observed[0][0] == (H, W, 4)
    assert observed[0][1] == "refinement_simple"
    assert observed[0][2]["relabel_margin"] == 0.25
    assert processed["output"].shape == (H * 2, W * 2)
    assert processed["output"].dtype == np.uint8
    assert processed["output_space"] == "preprocessed"


def test_halo_av_ignores_embedded_refinement_and_defaults_to_basic():
    ensemble = _DirectAV(
        _UnusedBackend(),
        _config(graph_refinement={"mode": "full", "relabel_margin": 999.0}),
    )
    full = ensemble.predict_step_full(_batch())
    assert full["refinement_mode"] == "basic"
    assert full["refinement_parameters"]["relabel_margin"] == 0.15


def test_halo_av_rejects_unknown_constructor_refinement_parameters():
    with pytest.raises(ValueError, match="Unknown AV refinement parameters"):
        _DirectAV(
            _UnusedBackend(),
            _config(),
            refinement_parameters={"not_a_parameter": 1},
        )


def test_deprecated_predict_batch_warns_and_preserves_prediction_geometry():
    ensemble = _DirectSegmentation(_UnusedBackend(), _config())
    with pytest.warns(FutureWarning, match="_predict_batch is deprecated"):
        items = ensemble._predict_batch(_batch())
    assert set(items[0]) == {"id", "image"}
    assert items[0]["image"].shape == (H, W, 3)
