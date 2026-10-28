from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from rtnls_inference.datasets import fundus
from rtnls_inference.datasets.fundus import FundusInferenceDataset


@pytest.mark.parametrize("field", ["mask", "labels", "keypoints", "logits", "unknown"])
@pytest.mark.parametrize("value", ["value", None])
def test_inference_rejects_unknown_and_training_fields(field, value):
    with pytest.raises(ValidationError, match=field):
        FundusInferenceDataset({"images": [{"id": "sample", field: value}]})


def test_records_are_validated_once_and_stored_as_dictionaries(monkeypatch):
    adapter = Mock(wraps=fundus.image_inputs_adapter)
    monkeypatch.setattr(fundus, "image_inputs_adapter", adapter)
    pixels = np.zeros((8, 12, 3), dtype=np.uint8)
    dataset = FundusInferenceDataset(
        {
            "images": [
                {
                    "id": np.int64(123),
                    "image": pixels,
                    "fov": np.float32(45),
                    "metadata": {"source": "study", "bounds": [1, 2, 3, 4]},
                }
            ]
        }
    )
    assert type(dataset.data[0]) is dict
    assert type(dataset.data[0]["metadata"]) is dict
    assert dataset.data[0]["id"] == "123"
    assert dataset.data[0]["fov"] == 45.0
    for _ in range(2):
        item = dataset[0]
        item["image"][:] = 100
    adapter.validate_python.assert_called_once()
    assert not pixels.any()


def test_loaded_tensors_become_cpu_hwc_arrays():
    tensor = torch.zeros(3, 8, 12, dtype=torch.uint8)
    record = FundusInferenceDataset({"images": [{"image": tensor}]}).data[0]
    assert isinstance(record["image"], np.ndarray)
    assert record["image"].shape == (8, 12, 3)
    assert record.get("id") is None


def test_path_loading_is_lazy_and_defaults_ids(tmp_path):
    path = tmp_path / "missing.jpg"
    dataset = FundusInferenceDataset(
        {"images": [{"image": path, "input_mask": Path("mask.png")}]}
    )
    assert dataset.data[0] == {
        "image": str(path),
        "input_mask": "mask.png",
        "id": "missing",
    }
    with pytest.raises(RuntimeError, match="missing"):
        dataset[0]


@pytest.mark.parametrize("fov", [True, "45", []])
def test_fov_must_be_numeric(fov):
    with pytest.raises(ValueError, match="fov"):
        FundusInferenceDataset({"images": [{"fov": fov}]})


def test_metadata_source_is_validated_and_extra_metadata_is_preserved():
    with pytest.raises(ValueError, match="metadata.source"):
        FundusInferenceDataset({"images": [{"metadata": {"source": 123}}]})
    record = FundusInferenceDataset(
        {"images": [{"metadata": {"source": None, "custom": [1, 2]}}]}
    ).data[0]
    assert record["metadata"] == {"custom": [1, 2]}


@pytest.mark.parametrize("data", [{}, {"images": [], "labels": {}}, {"images": None}])
def test_collection_structure_is_checked(data):
    with pytest.raises(ValueError):
        FundusInferenceDataset(data)
