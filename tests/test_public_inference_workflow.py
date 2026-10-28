from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from pydantic import ValidationError
from rtnls_fundusprep.preprocessor import FundusPreprocessor
from rtnls_inference.ensembles import (
    ClassificationEnsemble,
    EmbeddingEnsemble,
    HeatmapRegressionEnsemble,
    KeypointsEnsemble,
    RegressionEnsemble,
    SegmentationEnsemble,
    unbatch_predictions,
)


class VectorMembers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        values = image[:, :2].mean(dim=(-2, -1)) + self.anchor
        return torch.stack([values, values + 0.2])


class SpatialMembers(VectorMembers):
    def forward(self, image):
        self.calls += 1
        values = image[:, :2] + self.anchor
        return torch.stack([values - 0.2, values + 0.2])


FAMILIES = [
    RegressionEnsemble,
    ClassificationEnsemble,
    EmbeddingEnsemble,
    KeypointsEnsemble,
    HeatmapRegressionEnsemble,
    SegmentationEnsemble,
]


def make_ensemble(cls=RegressionEnsemble, **transform):
    backend = (
        SpatialMembers()
        if issubclass(cls, (SegmentationEnsemble, HeatmapRegressionEnsemble))
        else VectorMembers()
    )
    return cls(
        backend,
        {
            "datamodule": {
                "test_transform": {
                    "square_size": 12,
                    "resize": 8,
                    "contrast_enhance": False,
                    "normalize": "zero-one",
                    **transform,
                },
                "normalize_keypoints": True,
            },
            "lightningmodule": {"n_class": 2, "context_size": 12, "output_size": 8},
            "inference": {"tracing_input_size": [8, 8], "batch_size": 2, "tta": False},
        },
    ).eval()


@pytest.fixture
def data(tmp_path):
    y, x = np.mgrid[:12, :12]
    records = []
    for index in range(3):
        image = np.stack([x * 18, y * 18, 200 - x * 5], axis=-1).astype(np.uint8)
        image = np.roll(image, index, axis=0)
        path = tmp_path / f"input_{index}.png"
        Image.fromarray(image).save(path)
        records.append({"id": str(index), "image": str(path)})
    return records


@pytest.mark.parametrize("cls", FAMILIES)
def test_single_full_and_bulk_results_agree(cls, data, tmp_path, monkeypatch):
    ensemble = make_ensemble(cls)
    prepared = [
        ensemble.preprocess_item(row, input_space="preprocessed") for row in data
    ]
    expected = []
    for item in prepared:
        batch = ensemble.make_batch([item])
        calls = ensemble.ensemble.calls
        prediction = ensemble.predict_batch(batch)
        assert ensemble.ensemble.calls == calls + 1
        assert set(prediction) == {"aggregate", "prediction_size"}
        assert isinstance(prediction["aggregate"], torch.Tensor)
        assert not prediction["aggregate"].requires_grad
        expected.extend(ensemble.postprocess_batch(prediction, batch))
        assert (
            ensemble.ensemble.calls == calls + 1
        )  # Postprocessing never reruns inference.

    batch = ensemble.make_batch(prepared)
    full = ensemble.predict_batch(batch, fields="all")
    inspected = [
        ensemble.postprocess_item(item) for item in unbatch_predictions(full, batch)
    ]
    for single, complete in zip(expected, inspected):
        np.testing.assert_allclose(single["output"], complete["output"], atol=1e-6)
        assert single["geometry"] == complete["geometry"]

    predict = ensemble.predict_batch

    def no_inspection(batch, *, fields=None):
        assert fields is None
        return predict(batch)

    monkeypatch.setattr(ensemble, "predict_batch", no_inspection)
    loader = ensemble.make_inference_dataloader(
        data, input_space="preprocessed", batch_size=2
    )
    destination = tmp_path / "exports"
    streamed = list(ensemble.iter_predictions(loader))
    for result, single in zip(streamed, expected):
        np.testing.assert_allclose(result["output"], single["output"], atol=1e-6)
    manifest = ensemble.predict_dataloader(loader, destination)
    assert list(manifest["id"]) == [row["id"] for row in data]
    if not isinstance(ensemble, SegmentationEnsemble):
        np.testing.assert_allclose(
            manifest.drop(columns="id").to_numpy(),
            np.stack([item["output"].flatten() for item in expected]),
            atol=1e-6,
        )
        pd.testing.assert_frame_equal(ensemble.predict_dataloader(loader), manifest)
        saved = pd.read_csv(destination / "predictions.csv", dtype={"id": str})
        pd.testing.assert_frame_equal(saved, manifest, check_dtype=False)
        assert [path.name for path in destination.iterdir()] == ["predictions.csv"]
        return
    saved_manifest = pd.read_csv(destination / "manifest.csv", dtype={"id": str})
    assert list(saved_manifest["output_path"]) == list(manifest["output_path"])
    for row, single in zip(manifest.to_dict("records"), expected):
        path = Path(row["output_path"])
        saved = np.asarray(Image.open(path)) if path.suffix == ".png" else np.load(path)
        np.testing.assert_allclose(saved, single["output"], atol=1e-6)
        assert saved.dtype == single["output"].dtype


@pytest.mark.parametrize("num_workers", [0, 2])
def test_preparation_matches_loader_with_ce_masks_and_metadata(
    data, tmp_path, num_workers
):
    ensemble = make_ensemble(contrast_enhance=True)
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.full((12, 12), 255, dtype=np.uint8)).save(mask_path)
    records = [
        dict(row, contrast_enhanced=row["image"], input_mask=str(mask_path))
        for row in data
    ]
    records[0]["metadata"] = {"source": "first", "extra": [1, 2]}
    records[1]["metadata"] = {"source": "second", "different": "field"}
    prepared = [
        ensemble.preprocess_item(row, input_space="preprocessed") for row in records
    ]
    expected = ensemble.make_batch(prepared)
    loader = ensemble.make_inference_dataloader(
        records,
        input_space="preprocessed",
        batch_size=3,
        num_workers=num_workers,
    )
    actual = next(iter(loader))
    assert actual["image"].shape == (3, 6, 8, 8)
    assert actual["image"].device.type == "cpu"
    assert actual["preprocessed_image"].shape == (3, 12, 12, 3)
    assert actual["metadata"] == expected["metadata"]
    for field in ("image", "preprocessed_image", "input_mask"):
        torch.testing.assert_close(actual[field], expected[field])


@pytest.mark.parametrize("contrast_enhance", [False, True])
def test_raw_preprocessing_and_preprocessed_mode(tmp_path, contrast_enhance):
    y, x = np.mgrid[:120, :160]
    circle = (x - 80) ** 2 + (y - 60) ** 2 < 50**2
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[circle] = [160, 90, 60]
    path = tmp_path / "raw.png"
    Image.fromarray(image).save(path)
    record = {"id": "raw", "image": str(path)}
    ensemble = make_ensemble(contrast_enhance=contrast_enhance)
    raw = ensemble.preprocess_item(record, input_space="raw")
    loader = ensemble.make_inference_dataloader([record], input_space="raw")
    actual = next(iter(loader))
    torch.testing.assert_close(actual["image"], raw["image"][None])
    assert raw["preprocessed_image"].shape == (12, 12, 3)
    assert "bounds" in raw["metadata"]
    preprocessed = ensemble.preprocess_item(record, input_space="preprocessed")
    assert preprocessed["preprocessed_image"].shape == image.shape
    channels = 6 if contrast_enhance else 3
    assert preprocessed["image"].shape == raw["image"].shape == (channels, 8, 8)
    assert not torch.equal(preprocessed["image"], raw["image"])
    if contrast_enhance:
        reference = FundusPreprocessor(square_size=12, contrast_enhance=True)(
            image=image
        )
        expected_ce = cv2.resize(reference["ce"], (8, 8)).transpose(2, 0, 1)
        torch.testing.assert_close(
            raw["image"][3:], torch.from_numpy(expected_ce).float() / 255
        )


@pytest.mark.parametrize(
    "cls",
    [
        RegressionEnsemble,
        ClassificationEnsemble,
        EmbeddingEnsemble,
        KeypointsEnsemble,
        HeatmapRegressionEnsemble,
    ],
)
def test_caller_table_adaptation_keeps_values_and_does_not_write(cls, data, tmp_path):
    ensemble = make_ensemble(cls)
    loader = ensemble.make_inference_dataloader(
        data, input_space="preprocessed", batch_size=2
    )
    expected = np.stack(
        [item["output"].flatten() for item in ensemble.iter_predictions(loader)]
    )
    table = ensemble.predict_dataloader(loader)
    adapted = table.set_index("id")
    if not issubclass(cls, KeypointsEnsemble):
        adapted = adapted.filter(regex=f"^{ensemble.prediction_prefix}_")
        adapted.columns = range(len(adapted.columns))
    adapted.index.name = None
    np.testing.assert_allclose(adapted.to_numpy(), expected, atol=1e-6)
    assert list(adapted.index) == [row["id"] for row in data]
    assert not list(tmp_path.glob("predictions.csv"))
    empty = ensemble.predict_dataloader(ensemble.make_inference_dataloader([]))
    assert empty.empty and "id" in empty.columns


def test_skipped_and_empty_inputs_have_consistent_bulk_behavior(data, tmp_path):
    ensemble = make_ensemble()
    missing = {"id": "missing", "image": str(tmp_path / "absent.png")}
    strict = ensemble.make_inference_dataloader([missing], input_space="preprocessed")
    with pytest.raises(RuntimeError, match="missing"):
        next(iter(strict))
    loader = ensemble.make_inference_dataloader(
        [missing, data[0]],
        input_space="preprocessed",
        batch_size=1,
        ignore_exceptions=True,
    )
    with pytest.warns(UserWarning, match="Exception with image missing"):
        manifest = ensemble.predict_dataloader(loader, tmp_path / "skipped")
    assert list(manifest["id"]) == ["0"]
    empty_loader = ensemble.make_inference_dataloader([], input_space="preprocessed")
    empty = ensemble.predict_dataloader(empty_loader, tmp_path / "empty")
    assert empty.empty
    assert list(empty.columns) == list(manifest.columns)
    assert (tmp_path / "empty" / "predictions.csv").exists()


def test_export_rejects_duplicates_and_requires_explicit_overwrite(data, tmp_path):
    ensemble = make_ensemble()
    loader = ensemble.make_inference_dataloader(data, input_space="preprocessed")
    path = tmp_path / "exports"
    first = ensemble.predict_dataloader(loader, path)
    with pytest.raises(FileExistsError):
        ensemble.predict_dataloader(loader, path)
    again = ensemble.predict_dataloader(loader, path, overwrite=True)
    pd.testing.assert_frame_equal(first, again)
    with pytest.raises(ValueError, match="Duplicate input id"):
        ensemble.make_inference_dataloader(
            [data[0], data[0]], input_space="preprocessed"
        )


def test_public_loader_preserves_subclass_override(data):
    class RestrictedEnsemble(RegressionEnsemble):
        def make_inference_dataloader(self, inputs, batch_size=None, **kwargs):
            if batch_size not in (None, 1):
                raise ValueError("single image only")
            return super().make_inference_dataloader(inputs, batch_size=1, **kwargs)

    ensemble = RestrictedEnsemble(VectorMembers(), make_ensemble().config)
    loader = ensemble.make_inference_dataloader(data, input_space="preprocessed")
    assert loader.batch_size == 1
    with pytest.raises(ValueError, match="single image only"):
        ensemble.make_inference_dataloader(
            data, input_space="preprocessed", batch_size=2
        )


def test_invalid_input_space_and_mismatched_prediction_batch(data):
    ensemble = make_ensemble()
    with pytest.raises(ValueError, match="input_space"):
        ensemble.preprocess_item(data[0], input_space="unknown")
    with pytest.raises(ValueError, match="input_space"):
        ensemble.make_inference_dataloader(data, input_space="unknown")
    with pytest.raises(ValidationError):
        ensemble.preprocess_item(dict(data[0], labels=[1]))
    batch = next(
        iter(ensemble.make_inference_dataloader(data, input_space="preprocessed"))
    )
    prediction = ensemble.predict_batch(batch)
    with pytest.raises(ValueError, match="batch sizes differ"):
        ensemble.postprocess_batch(prediction, {"image": batch["image"][:1]})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_predict_batch_retains_device_until_postprocessing(data, monkeypatch):
    from rtnls_inference.ensembles import predict_output

    ensemble = make_ensemble().to("cuda")
    batch = next(
        iter(ensemble.make_inference_dataloader(data, input_space="preprocessed"))
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            predict_output,
            "to_numpy",
            lambda value: pytest.fail("Unexpected CPU conversion"),
        )
        prediction = ensemble.predict_batch(batch)
    assert prediction["aggregate"].device.type == "cuda"
    assert batch["image"].device.type == "cpu"
    results = ensemble.postprocess_batch(prediction, batch)
    assert isinstance(results[0]["output"], np.ndarray)


@pytest.mark.parametrize("contrast_enhance", [False, True])
@pytest.mark.parametrize("kind", ["numpy", "tensor"])
def test_loaded_raw_images_match_paths(tmp_path, kind, contrast_enhance):
    y, x = np.mgrid[:120, :160]
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    circle = (x - 80) ** 2 + (y - 60) ** 2 < 50**2
    image[circle] = [160, 90, 60]
    path = tmp_path / "fundus.png"
    Image.fromarray(image).save(path)
    loaded = torch.from_numpy(image).permute(2, 0, 1) if kind == "tensor" else image
    ensemble = make_ensemble(SegmentationEnsemble, contrast_enhance=contrast_enhance)
    from_path = ensemble.preprocess_item({"image": path})
    from_memory = ensemble.preprocess_item({"image": loaded})
    assert from_path["id"] == "fundus"
    assert from_memory["id"] is None
    assert from_path["metadata"] == from_memory["metadata"]
    torch.testing.assert_close(from_path["image"], from_memory["image"])
    np.testing.assert_array_equal(
        from_path["preprocessed_image"], from_memory["preprocessed_image"]
    )
    batch = ensemble.make_batch([from_path, from_memory])
    results = ensemble.postprocess_batch(ensemble.predict_batch(batch), batch)
    assert [item["id"] for item in results] == ["fundus", None]
    np.testing.assert_array_equal(results[0]["output"], results[1]["output"])
    assert results[0]["geometry"] == results[1]["geometry"]
    np.testing.assert_array_equal(image, np.array(Image.open(path)))


@pytest.mark.parametrize("cls", FAMILIES)
@pytest.mark.parametrize("num_workers", [0, 2])
def test_mixed_loaded_inputs_and_anonymous_batches(cls, data, num_workers):
    image = np.array(Image.open(data[0]["image"]))
    tensor = torch.from_numpy(image).permute(2, 0, 1)
    records = [
        {"image": data[0]["image"]},
        {"image": image},
        {"image": tensor},
        {"id": "named", "image": image},
    ]
    ensemble = make_ensemble(cls)
    loader = ensemble.make_inference_dataloader(
        records, input_space="preprocessed", batch_size=4, num_workers=num_workers
    )
    batch = next(iter(loader))
    assert batch["id"] == ["input_0", None, None, "named"]
    assert batch["image"].device.type == "cpu"
    results = ensemble.postprocess_batch(ensemble.predict_batch(batch), batch)
    assert [item["id"] for item in results] == batch["id"]
    for result in results[1:]:
        np.testing.assert_allclose(result["output"], results[0]["output"])


@pytest.mark.parametrize("identifier", [None, "", "   "])
def test_missing_ids_default_only_for_paths(identifier, data):
    ensemble = make_ensemble()
    image = np.array(Image.open(data[0]["image"]))
    for source, expected in ((Path(data[0]["image"]), "input_0"), (image, None)):
        item = ensemble.preprocess_item(
            {"image": source, "id": identifier}, input_space="preprocessed"
        )
        assert item["id"] == expected


@pytest.mark.parametrize(
    "image, message",
    [
        (np.zeros((8, 8, 3), dtype=np.float32), "RGB uint8"),
        (torch.zeros((3, 8, 8)), "RGB uint8"),
        (np.zeros((3, 8, 8), dtype=np.uint8), r"\(H, W, 3\)"),
        (torch.zeros((8, 8, 3), dtype=torch.uint8), r"\(3, H, W\)"),
        (np.zeros((1, 8, 8, 3), dtype=np.uint8), r"\(H, W, 3\)"),
        (np.zeros((0, 8, 3), dtype=np.uint8), r"\(H, W, 3\)"),
    ],
)
def test_invalid_loaded_image_format_is_rejected_before_workers(image, message):
    with pytest.raises(ValueError, match=message):
        make_ensemble().make_inference_dataloader(
            [{"image": image}], num_workers=2, ignore_exceptions=True
        )


def test_path_records_stay_serializable_and_loaded_records_are_supported(data):
    import json
    from rtnls_inference.datasets.fundus import FundusInferenceDataset

    record = FundusInferenceDataset(
        {"images": [{"image": Path(data[0]["image"])}]}
    ).data[0]
    assert json.loads(json.dumps(record))["id"] == "input_0"
    image = np.array(Image.open(data[0]["image"]))
    record = {"image": torch.from_numpy(image).permute(2, 0, 1)}
    item = make_ensemble().preprocess_item(record, input_space="preprocessed")
    assert item["id"] is None
    assert item["image"].shape == (3, 8, 8)


@pytest.mark.parametrize(
    "records",
    [
        [{"image": "/first/photo.jpg"}, {"image": "/second/photo.png"}],
        [{"id": "photo", "image": "other.jpg"}, {"image": "photo.jpg"}],
        [{"id": 1, "image": "first.jpg"}, {"id": "1", "image": "second.jpg"}],
    ],
)
def test_duplicate_ids_fail_before_dataloader_construction(records, monkeypatch):
    from rtnls_inference.ensembles import base

    ensemble = make_ensemble()
    monkeypatch.setattr(
        base, "DataLoader", lambda *args, **kwargs: pytest.fail("loader constructed")
    )
    with pytest.raises(ValueError, match="Duplicate input id"):
        ensemble.make_inference_dataloader(records, ignore_exceptions=True)
    assert ensemble.ensemble.calls == 0


@pytest.mark.parametrize("method", ["predict_batch", "predict_step"])
def test_manual_batches_reject_duplicates_before_prediction(method):
    ensemble = make_ensemble()
    with pytest.raises(ValueError, match="Duplicate input id"):
        getattr(ensemble, method)({"image": torch.zeros(2, 3, 8, 8), "id": [1, "1"]})
    assert ensemble.ensemble.calls == 0
    with pytest.raises(ValueError, match="Duplicate input id"):
        ensemble.make_batch(
            [
                {"image": torch.zeros(3, 8, 8), "id": "same"},
                {"image": torch.zeros(3, 8, 8), "id": "same"},
            ]
        )


@pytest.mark.parametrize(
    "identifier, message",
    [(None, "requires an id"), ("../outside", "filename component")],
)
def test_invalid_export_ids_fail_before_prediction_or_writing(
    data, tmp_path, identifier, message
):
    ensemble = make_ensemble(SegmentationEnsemble)
    image = np.array(Image.open(data[0]["image"]))
    loader = ensemble.make_inference_dataloader(
        [{"image": data[0]["image"]}, {"image": image, "id": identifier}],
        input_space="preprocessed",
        batch_size=1,
    )
    method = ensemble.predict_dataloader
    with pytest.raises(ValueError, match=message):
        method(loader, tmp_path / "exports")
    assert ensemble.ensemble.calls == 0
    assert not (tmp_path / "exports").exists()


def test_loaded_images_export_with_explicit_ids(data, tmp_path):
    image = np.array(Image.open(data[0]["image"]))
    records = [
        {"id": "numpy", "image": image},
        {"id": "tensor", "image": torch.from_numpy(image).permute(2, 0, 1)},
        {"image": data[0]["image"]},
    ]
    ensemble = make_ensemble(SegmentationEnsemble)
    loader = ensemble.make_inference_dataloader(records, input_space="preprocessed")
    manifest = ensemble.predict_dataloader(loader, tmp_path / "exports")
    assert list(manifest["id"]) == ["numpy", "tensor", "input_0"]
    assert all(Path(path).is_file() for path in manifest["output_path"])
    masks = [np.array(Image.open(path)) for path in manifest["output_path"]]
    np.testing.assert_array_equal(masks[0], masks[1])
    np.testing.assert_array_equal(masks[0], masks[2])


@pytest.mark.parametrize(
    "cls",
    [
        RegressionEnsemble,
        ClassificationEnsemble,
        EmbeddingEnsemble,
        KeypointsEnsemble,
        HeatmapRegressionEnsemble,
    ],
)
def test_in_memory_predictions_do_not_require_ids(cls, data, tmp_path):
    image = np.array(Image.open(data[0]["image"]))
    ensemble = make_ensemble(cls)
    loader = ensemble.make_inference_dataloader(
        [{"image": image}, {"image": image}], input_space="preprocessed"
    )
    result = ensemble.predict_dataloader(loader)
    assert list(result["id"]) == [None, None]
    assert len(result) == 2
    assert not (tmp_path / "unused").exists()


def test_dataframe_records_default_ids_to_filename_or_none(data):
    ensemble = make_ensemble()
    frame = pd.DataFrame({"image": [row["image"] for row in data]})
    loader = ensemble.make_inference_dataloader(
        frame.to_dict("records"), input_space="preprocessed"
    )
    assert list(ensemble.predict_dataloader(loader)["id"]) == [
        "input_0",
        "input_1",
        "input_2",
    ]
    image = np.array(Image.open(data[0]["image"]))
    loader = ensemble.make_inference_dataloader(
        [{"image": image}], input_space="preprocessed"
    )
    assert list(ensemble.predict_dataloader(loader)["id"]) == [None]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_loaded_cuda_image_is_converted_before_workers(data):
    image = np.array(Image.open(data[0]["image"]))
    tensor = torch.from_numpy(image).permute(2, 0, 1).to("cuda")
    ensemble = make_ensemble().to("cuda")
    loader = ensemble.make_inference_dataloader(
        [{"image": tensor}], input_space="preprocessed", num_workers=2
    )
    assert isinstance(loader.dataset.data[0]["image"], np.ndarray)
    batch = next(iter(loader))
    assert batch["image"].device.type == "cpu"
    results = ensemble.postprocess_batch(ensemble.predict_batch(batch), batch)
    assert results[0]["id"] is None
    np.testing.assert_array_equal(tensor.cpu().permute(1, 2, 0).numpy(), image)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
@pytest.mark.parametrize("method", ["predict_batch", "predict_step"])
@pytest.mark.parametrize("fail", [False, True])
def test_inference_uses_model_device_and_restores_current_device(method, fail):
    caller_device = torch.cuda.current_device()
    model_device = (caller_device + 1) % torch.cuda.device_count()

    class DeviceCheckingMembers(VectorMembers):
        def forward(self, image):
            assert torch.cuda.current_device() == image.device.index == model_device
            if fail:
                raise RuntimeError("Prediction failed")
            return super().forward(image)

    ensemble = RegressionEnsemble(DeviceCheckingMembers(), make_ensemble().config).to(
        f"cuda:{model_device}"
    )
    batch = {"image": torch.zeros(1, 3, 8, 8)}
    if fail:
        with pytest.raises(RuntimeError, match="Prediction failed"):
            getattr(ensemble, method)(batch)
    else:
        result = getattr(ensemble, method)(batch)
        values = result if method == "predict_step" else result["aggregate"]
        assert values.shape == (1, 2)
    assert torch.cuda.current_device() == caller_device


@pytest.mark.parametrize("method", ["predict_batch", "predict_step"])
def test_cpu_inference_does_not_enter_cuda_device_context(method, monkeypatch):
    def unexpected_cuda_context(*args, **kwargs):
        raise AssertionError("CPU inference must not select a CUDA device")

    monkeypatch.setattr(torch.cuda, "device", unexpected_cuda_context)
    ensemble = make_ensemble()
    getattr(ensemble, method)({"image": torch.zeros(1, 3, 8, 8)})


@pytest.mark.parametrize("cls", FAMILIES)
def test_iterator_is_lazy_and_returns_one_batch_at_a_time(cls, data):
    ensemble = make_ensemble(cls)
    loader = ensemble.make_inference_dataloader(
        data, input_space="preprocessed", batch_size=1
    )
    results = ensemble.iter_predictions(loader)
    assert ensemble.ensemble.calls == 0
    first = next(results)
    assert ensemble.ensemble.calls == 1
    assert first["id"] == "0" and "preprocessed_image" in first
    assert next(results)["id"] == "1"
    assert ensemble.ensemble.calls == 2


@pytest.mark.parametrize(
    "cls",
    [
        RegressionEnsemble,
        ClassificationEnsemble,
        EmbeddingEnsemble,
        KeypointsEnsemble,
        HeatmapRegressionEnsemble,
    ],
)
def test_numeric_export_requires_ids_but_in_memory_tables_do_not(cls, data, tmp_path):
    ensemble = make_ensemble(cls)
    image = np.array(Image.open(data[0]["image"]))
    loader = ensemble.make_inference_dataloader(
        [{"image": image}], input_space="preprocessed"
    )
    with pytest.raises(ValueError, match="requires an id"):
        ensemble.predict_dataloader(loader, tmp_path / "outputs")
    assert ensemble.ensemble.calls == 0
    assert not (tmp_path / "outputs").exists()
    table = ensemble.predict_dataloader(loader)
    assert table["id"].tolist() == [None]
    assert "preprocessed_image" not in table


def test_table_collection_releases_image_context(monkeypatch):
    import weakref

    ensemble = make_ensemble()
    images = []

    def results(loader):
        for index in range(6):
            # The consumer may still hold the immediately preceding item.
            assert all(reference() is None for reference in images[:-1])
            image = np.zeros((128, 128, 3), dtype=np.uint8)
            images.append(weakref.ref(image))
            yield {
                "id": str(index),
                "output": np.array([index], dtype=np.float32),
                "preprocessed_image": image,
            }

    monkeypatch.setattr(ensemble, "iter_predictions", results)
    table = ensemble.predict_dataloader([])
    assert table["prediction_0"].tolist() == list(range(6))
    assert all(reference() is None for reference in images)


def test_segmentation_bulk_requires_destination(data, tmp_path):
    ensemble = make_ensemble(SegmentationEnsemble)
    loader = ensemble.make_inference_dataloader(data, input_space="preprocessed")
    with pytest.raises(ValueError, match="iter_predictions"):
        ensemble.predict_dataloader(loader)
    assert ensemble.ensemble.calls == 0


def test_numeric_summary_rejects_changed_output_width(monkeypatch):
    ensemble = make_ensemble()
    monkeypatch.setattr(
        ensemble,
        "iter_predictions",
        lambda loader: iter(
            [
                {"id": "a", "output": np.zeros(2)},
                {"id": "b", "output": np.zeros(3)},
            ]
        ),
    )
    with pytest.raises(ValueError, match="columns changed"):
        ensemble.predict_dataloader([])


def test_lightning_predict_loop_returns_aggregate_tensors():
    import lightning as L
    from torch.utils.data import DataLoader

    ensemble = make_ensemble(RegressionEnsemble)
    loader = DataLoader([{"image": torch.zeros(3, 8, 8)}] * 3, batch_size=2)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    outputs = trainer.predict(ensemble, dataloaders=loader)
    assert [tuple(value.shape) for value in outputs] == [(2, 2), (1, 2)]
    assert all(not value.requires_grad for value in outputs)
