from __future__ import annotations

import numpy as np
import pytest
import torch
from rtnls_inference.ensembles.ensemble_classification import ClassificationEnsemble
from rtnls_inference.ensembles.ensemble_embedding import EmbeddingEnsemble
from rtnls_inference.ensembles.ensemble_heatmap_regression import (
    HeatmapRegressionEnsemble,
)
from rtnls_inference.ensembles.ensemble_keypoints import KeypointsEnsemble
from rtnls_inference.ensembles.ensemble_regression import RegressionEnsemble
from rtnls_inference.ensembles.ensemble_segmentation import (
    SegmentationEnsemble,
)
from rtnls_inference.ensembles.predict_output import (
    unbatch_predictions,
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
        (RegressionEnsemble(_MemberFirstBackend(4), _config()), (N, M, 4), (N, 4)),
        (ClassificationEnsemble(_MemberFirstBackend(3), _config()), (N, M, 3), (N, 3)),
        (EmbeddingEnsemble(_MemberFirstBackend(5), _config()), (N, M, 5), (N, 5)),
        (KeypointsEnsemble(_MemberFirstBackend(4), _config()), (N, M, 2, 2), (N, 2, 2)),
        (_DirectHeatmap(_UnusedBackend(), _config()), (N, M, 2, 2), (N, 2, 2)),
    ],
)
def test_family_contracts_are_batch_major_and_aggregated(
    ensemble, member_shape, aggregate_shape
):
    batch = _batch()
    aggregate = ensemble.predict_step(batch)
    full = ensemble.predict_batch(batch, fields="all")

    assert aggregate.shape == aggregate_shape
    assert aggregate.device == batch["image"].device
    assert not aggregate.requires_grad
    assert full["prediction"].shape == member_shape
    assert full["aggregate"].shape == aggregate_shape
    assert isinstance(full, dict)
    assert isinstance(full["prediction"], torch.Tensor)
    assert isinstance(full["aggregate"], torch.Tensor)
    assert unbatch_predictions(full, batch)[0]["preprocessed_image"].shape == (
        H * 2,
        W * 2,
        3,
    )
    np.testing.assert_allclose(full["aggregate"], aggregate.numpy(), rtol=1e-5)


def test_spatial_intermediates_are_validated():
    segmentation = _DirectSegmentation(_UnusedBackend(), _config())
    segmentation_full = segmentation.predict_batch(_batch(), fields="all")
    assert segmentation_full["logits"].shape == (N, H, W, 3)

    heatmap = _DirectHeatmap(_UnusedBackend(), _config()).predict_batch(
        _batch(), fields="all"
    )
    assert heatmap["heatmaps"].shape == (N, M, 2, H, W)


def test_scalar_regression_is_promoted_to_nc_and_nmc():
    ensemble = RegressionEnsemble(_ScalarRegressionBackend(), _config())
    aggregate = ensemble.predict_step(_batch())
    full = ensemble.predict_batch(_batch(), fields="all")
    assert aggregate.shape == (N, 1)
    assert full["prediction"].shape == (N, M, 1)


def test_direct_tensor_batch_uses_model_input_as_canonical_fallback():
    ensemble = _DirectSegmentation(_UnusedBackend(), _config())
    full = ensemble.predict_batch(_batch(with_context=False), fields="all")
    assert "preprocessed_image" not in full
    assert [
        item["geometry"]
        for item in unbatch_predictions(full, _batch(with_context=False))
    ] == [{"preprocessed_size": (H, W), "prediction_size": (H, W)}] * N


@pytest.mark.parametrize("method", ["predict_batch", "predict_step"])
@pytest.mark.parametrize(
    "output, error, message",
    [
        (
            torch.zeros(N, M, H, W),
            ValueError,
            "segmentation prediction must have rank 5",
        ),
        (
            torch.zeros(N + 1, M, 2, H, W),
            ValueError,
            "Prediction and input batch sizes differ",
        ),
        (torch.zeros(N, 0, 2, H, W), ValueError, "member axes"),
        (torch.zeros(N, M, 2, H, W, dtype=torch.int64), TypeError, "floating point"),
        (np.zeros((N, M, 2, H, W)), TypeError, "torch.Tensor"),
    ],
)
def test_invalid_backend_outputs_fail(method, output, error, message, monkeypatch):
    ensemble = _DirectSegmentation(_UnusedBackend(), _config())
    monkeypatch.setattr(ensemble, "forward", lambda images: output)
    with pytest.raises(error, match=message):
        getattr(ensemble, method)(_batch())


def test_canonical_context_batch_length_is_checked():
    ensemble = _DirectSegmentation(_UnusedBackend(), _config())
    batch = _batch()
    batch["preprocessed_image"] = batch["preprocessed_image"][:1]
    with pytest.raises(ValueError, match="preprocessed_image batch size"):
        unbatch_predictions(ensemble.predict_batch(batch, fields="all"), batch)


def test_plain_output_and_decollation_preserve_ndarrays():
    full = _DirectSegmentation(_UnusedBackend(), _config()).predict_batch(
        _batch(), fields="all"
    )
    items = unbatch_predictions(full, _batch())
    assert len(items) == N
    assert isinstance(items[0]["prediction"], np.ndarray)
    assert items[0]["prediction"].shape == (M, H, W, 3)
    assert items[1]["id"] == "second"


def test_resize_only_restoration_and_non_square_point_scaling():
    geometry = {"preprocessed_size": (8, 12), "prediction_size": (4, 3)}
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
    "geometry",
    [
        {"preprocessed_size": (0, 12)},
        {"preprocessed_size": (8, 12), "prediction_size": (-1, 3)},
        {"preprocessed_size": (8.5, 12)},
        {"preprocessed_size": (8,)},
    ],
)
def test_invalid_geometry_reports_the_bad_dimensions(geometry):
    with pytest.raises(ValueError, match="positive integer dimensions"):
        restore_array_to_preprocessed(np.zeros((4, 3)), geometry)


def test_wrong_channel_count_and_invalid_keypoints_report_clear_errors(monkeypatch):
    ensemble = _DirectSegmentation(
        _UnusedBackend(), {**_config(), "lightningmodule": {"n_class": 4}}
    )
    with pytest.raises(ValueError, match="Expected 4 segmentation channels"):
        ensemble.predict_batch(_batch())
    keypoints = KeypointsEnsemble(_MemberFirstBackend(3), _config())
    with pytest.raises(ValueError, match="xy pairs"):
        keypoints.predict_batch(_batch())


def test_aggregate_shape_is_checked(monkeypatch):
    ensemble = _DirectSegmentation(_UnusedBackend(), _config())
    monkeypatch.setattr(
        ensemble,
        "_aggregate_tensors",
        lambda prediction: prediction.mean(dim=1)[..., :1],
    )
    with pytest.raises(
        ValueError, match="prediction and aggregate output shapes differ"
    ):
        ensemble.predict_batch(_batch())


@pytest.mark.parametrize(
    "cls", [RegressionEnsemble, EmbeddingEnsemble, KeypointsEnsemble]
)
def test_vector_backend_requires_floating_point_predictions(cls, monkeypatch):
    backend = _MemberFirstBackend(4)
    monkeypatch.setattr(
        backend, "forward", lambda image: torch.zeros(M, N, 4, dtype=torch.int64)
    )
    ensemble = cls(backend, _config())
    with pytest.raises(TypeError, match="floating point.*torch.int64"):
        ensemble.predict_step(_batch())


def test_geometry_restoration_requires_arrays():
    geometry = {"preprocessed_size": (4, 6), "prediction_size": (4, 6)}
    with pytest.raises(TypeError, match="array must be a numpy.ndarray"):
        restore_array_to_preprocessed([[1, 2]], geometry)
    with pytest.raises(TypeError, match="points must be a numpy.ndarray"):
        restore_points_to_preprocessed([[1, 2]], geometry)
    with pytest.raises(ValueError, match="array dimensions must be positive"):
        restore_array_to_preprocessed(np.zeros((0, 6)), geometry)


@pytest.mark.parametrize("cls", [SegmentationEnsemble, HeatmapRegressionEnsemble])
@pytest.mark.parametrize(
    "kind, error, message",
    [
        ("array", TypeError, "tile prediction.*torch.Tensor"),
        ("rank", ValueError, "tile prediction.*rank 5"),
        ("batch", ValueError, "tile prediction batch size"),
        ("members", ValueError, "tile prediction.*positive"),
        ("dtype", TypeError, "tile prediction.*floating point"),
    ],
)
def test_tiled_inference_rejects_malformed_backend_outputs(cls, kind, error, message):
    class Backend(torch.nn.Module):
        def forward(self, image):
            shape = (1, len(image), 2, *image.shape[-2:])
            if kind == "array":
                return np.zeros(shape)
            if kind == "rank":
                return torch.zeros(shape[1:])
            if kind == "batch":
                return torch.zeros(1, len(image) + 1, *shape[2:])
            if kind == "members":
                return torch.zeros(0, *shape[1:])
            return torch.zeros(shape, dtype=torch.int64)

    ensemble = cls(Backend(), _config(tracing_input_size=[4, 6]))
    with pytest.raises(error, match=message):
        ensemble.predict_batch(_batch())


@pytest.mark.parametrize(
    "fields", [None, (), ("logits",), ("prediction", "logits"), "all"]
)
def test_field_selection_runs_once_and_keeps_required_fields(fields, monkeypatch):
    ensemble = _DirectSegmentation(_UnusedBackend(), _config())
    forward = ensemble.forward
    calls = []

    def counted(image):
        calls.append(image)
        return forward(image)

    monkeypatch.setattr(ensemble, "forward", counted)
    batch = _batch()
    result = ensemble.predict_batch(batch, fields=fields)
    assert len(calls) == 1
    expected = {"aggregate", "prediction_size"}
    expected.update(("prediction", "logits") if fields == "all" else (fields or ()))
    assert set(result) == expected
    for value in result.values():
        if isinstance(value, torch.Tensor):
            assert value.device == ensemble.get_device()
            assert not value.requires_grad
    items = unbatch_predictions(result, batch)
    assert items[1]["id"] == "second"
    assert items[0]["aggregate"].shape == (H, W, 3)
    assert items[0]["geometry"]["preprocessed_size"] == (H * 2, W * 2)
    assert len(calls) == 1


@pytest.mark.parametrize("fields", ["logits", 1, {"logits"}, [None], [3]])
def test_invalid_field_selection_has_a_clear_error(fields):
    ensemble = _DirectSegmentation(_UnusedBackend(), _config())
    with pytest.raises(TypeError, match="fields"):
        ensemble.predict_batch(_batch(), fields=fields)


def test_unavailable_field_names_report_available_fields():
    ensemble = _DirectSegmentation(_UnusedBackend(), _config())
    with pytest.raises(ValueError, match="heatmaps.*available fields.*logits"):
        ensemble.predict_batch(_batch(), fields=("heatmaps",))


@pytest.mark.parametrize("count", [1, 2])
def test_unbatch_preserves_shared_metadata_and_does_not_mutate_inputs(count):
    batch = {"id": [str(i) for i in range(count)], "image": torch.zeros(count, 3, 2, 2)}
    config = {"thresholds": list(range(count)), "nested": {"names": ["a", "b"]}}
    prediction = {
        "aggregate": torch.ones(count, 2, 2, 3, dtype=torch.float16),
        "prediction_size": (2, 2),
        "logit_names": ["a", "b"],
        "refinement_parameters": config,
    }
    items = unbatch_predictions(prediction, batch)
    assert isinstance(prediction["aggregate"], torch.Tensor)
    assert prediction["prediction_size"] == (2, 2)
    assert batch["image"].shape == (count, 3, 2, 2)
    for index, item in enumerate(items):
        assert item["id"] == str(index)
        assert item["refinement_parameters"] == config
        assert item["logit_names"] == ["a", "b"]
        assert item["aggregate"].dtype == np.float16
        assert item["geometry"] == {
            "preprocessed_size": (2, 2),
            "prediction_size": (2, 2),
        }


def test_unbatch_converts_bfloat16_and_checks_inspection_batch_axes():
    batch = _batch()
    result = {
        "aggregate": torch.ones(N, 3, dtype=torch.bfloat16),
        "prediction_size": None,
    }
    assert unbatch_predictions(result, batch)[0]["aggregate"].dtype == np.float32
    result["logits"] = torch.ones(N + 1, 3)
    with pytest.raises(ValueError, match="logits.*batch axis 2.*shape"):
        unbatch_predictions(result, batch)
    result["logits"] = np.array(1)
    with pytest.raises(ValueError, match="logits.*shape"):
        unbatch_predictions(result, batch)


def test_postprocessing_does_not_convert_inspection_only_tensors(monkeypatch):
    import rtnls_inference.ensembles.predict_output as output_module

    ensemble = _DirectSegmentation(_UnusedBackend(), _config())
    batch = _batch()
    result = ensemble.predict_batch(batch, fields="all")
    original = output_module.to_numpy
    extra = result["prediction"]

    def convert(value):
        assert value is not extra
        return original(value)

    monkeypatch.setattr(output_module, "to_numpy", convert)
    processed = ensemble.postprocess_batch(result, batch)
    assert processed[0]["output"].shape == (H * 2, W * 2)
    assert result["prediction"] is extra
