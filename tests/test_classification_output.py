import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from rtnls_inference.ensembles import ClassificationEnsemble, unbatch_predictions


class Members(torch.nn.Module):
    def forward(self, image):
        logits = torch.tensor(
            [[10.0, 0.0, -2.0], [0.0, 2.0, -2.0]], device=image.device
        )
        return logits[:, None, :].expand(-1, image.shape[0], -1)


def ensemble(metadata=None):
    config = {"datamodule": {}, "lightningmodule": {"n_class": 3}, "inference": {}}
    if metadata is not None:
        config["inference"]["classification"] = metadata
    return ClassificationEnsemble(Members(), config).eval()


def batch(ids=("first", "second")):
    return {"id": list(ids), "image": torch.zeros(len(ids), 3, 8, 8)}


METADATA = {"task": "multiclass", "label_map": {"0": "good", "1": "usable", "2": "bad"}}


def test_multiclass_decodes_mean_logits_without_changing_numeric_contracts():
    model = ensemble(deepcopy(METADATA))
    inputs = batch()
    predictions = model.predict_batch(inputs)
    expected = torch.tensor([[5.0, 1.0, -2.0], [5.0, 1.0, -2.0]])
    torch.testing.assert_close(predictions["aggregate"], expected)
    results = model.postprocess_batch(predictions, inputs)
    full = model.predict_batch(inputs, fields="all")
    full_items = unbatch_predictions(full, inputs)
    for result, full_item in zip(results, full_items):
        np.testing.assert_array_equal(result["output"], [5.0, 1.0, -2.0])
        assert result["output"].dtype == np.float32
        assert result["class_id"] == 0
        assert result["label"] == "good"
        np.testing.assert_allclose(
            result["probabilities"], expected.softmax(-1)[0], rtol=1e-6
        )
        np.testing.assert_array_equal(
            model.postprocess_item(full_item)["probabilities"], result["probabilities"]
        )
    assert not np.allclose(
        results[0]["probabilities"],
        model.ensemble(inputs["image"]).softmax(-1).mean(0)[0],
    )
    torch.testing.assert_close(model.predict_step(inputs), expected)
    np.testing.assert_array_equal(
        model.predict_dataloader([inputs]).filter(regex="^logit_").to_numpy(), expected
    )


def test_old_release_keeps_numeric_only_behavior(tmp_path):
    model = ensemble()
    inputs = batch()
    result = model.postprocess_batch(model.predict_batch(inputs), inputs)[0]
    assert "label" not in result and "probabilities" not in result
    model.predict_dataloader([inputs], tmp_path)
    table = pd.read_csv(tmp_path / "predictions.csv")
    assert list(table) == ["id", "logit_0", "logit_1", "logit_2"]
    assert [path.name for path in tmp_path.iterdir()] == ["predictions.csv"]


def test_task_without_names_returns_indices_and_probabilities():
    model = ensemble({"task": "multiclass"})
    inputs = batch((None,))
    result = model.postprocess_batch(model.predict_batch(inputs), inputs)[0]
    assert result["id"] is None and result["label"] is None
    assert result["class_id"] == 0


@pytest.mark.parametrize(
    "metadata",
    [
        {"task": "unknown"},
        {"label_map": {0: "good"}},
        {"task": "multiclass", "label_map": {1: "bad"}},
        {"task": "multiclass", "label_map": {0: "a", 1: "a"}},
        {"task": "multiclass", "label_map": {0: " "}},
        {"task": "multiclass", "label_map": {}},
        {"task": "multiclass", "label_map": {0: "a", "0": "b"}},
        {"task": "multiclass", "label_map": {0.5: "a", 1: "b", 2: "c"}},
        {"task": "multiclass", "label_map": {0: None, 1: "b", 2: "c"}},
        {"task": "multiclass", "label_map": {0: "a", 1: "b"}},
        {"task": "multiclass", "threshold": 0.5},
        {"task": "multilabel", "threshold": float("nan")},
        {"task": "multilabel", "threshold": 1.1},
        {"task": "multilabel", "threshold": []},
        {"task": "multilabel", "threshold": [[0.4, 0.5, 0.6]]},
        {"task": "multilabel", "threshold": [0.4, 0.5]},
    ],
)
def test_invalid_metadata_is_rejected_when_constructing_ensemble(metadata):
    with pytest.raises(ValueError):
        ensemble(metadata)


def test_label_count_is_checked_against_backend_output():
    model = ensemble(METADATA)
    inputs = batch()
    predictions = model.predict_batch(inputs)
    predictions["aggregate"] = predictions["aggregate"][:, :2]
    with pytest.raises(ValueError, match="all 2 output channels"):
        model.postprocess_batch(predictions, inputs)


def test_multilabel_uses_independent_sigmoids_and_thresholds(tmp_path):
    model = ensemble(
        {
            "task": "multilabel",
            "label_map": METADATA["label_map"],
            "threshold": [0.5, 0.9, 0.1],
        }
    )
    inputs = batch()
    predictions = model.predict_batch(inputs)
    result = model.postprocess_batch(predictions, inputs)[0]
    np.testing.assert_allclose(
        result["probabilities"], predictions["aggregate"].sigmoid()[0], rtol=1e-6
    )
    assert result["class_ids"] == [0, 2]
    assert result["labels"] == ["good", "bad"]
    assert "label" not in result
    returned = model.predict_dataloader([inputs], tmp_path)
    assert returned.loc[0, "labels"] == ["good", "bad"]
    assert returned.loc[0, "class_ids"] == [0, 2]
    table = pd.read_csv(tmp_path / "predictions.csv")
    assert json.loads(table.loc[0, "labels"]) == ["good", "bad"]
    assert json.loads(table.loc[0, "class_ids"]) == [0, 2]


def test_multilabel_threshold_boundary_and_empty_selection():
    model = ensemble({"task": "multilabel", "label_map": METADATA["label_map"]})
    inputs = batch((None,))
    result = model.postprocess_batch(
        {"aggregate": torch.zeros(1, 3), "prediction_size": None}, inputs
    )[0]
    assert result["class_ids"] == [] and result["labels"] == []


def test_bulk_returns_predictions_and_exports_only_the_summary(tmp_path):
    model = ensemble(METADATA)
    inputs = batch()
    returned = model.predict_dataloader([inputs], tmp_path)
    assert list(returned["id"]) == ["first", "second"]
    pd.testing.assert_frame_equal(model.predict_dataloader([inputs]), returned)
    assert [path.name for path in tmp_path.iterdir()] == ["predictions.csv"]
    table = pd.read_csv(tmp_path / "predictions.csv")
    assert list(table.columns) == [
        "id",
        "class_id",
        "label",
        "probability_good",
        "probability_usable",
        "probability_bad",
        "logit_good",
        "logit_usable",
        "logit_bad",
    ]
    assert list(table["label"]) == ["good", "good"]
    np.testing.assert_allclose(table.filter(like="probability_").sum(axis=1), 1.0)
    pd.testing.assert_frame_equal(table, returned, check_dtype=False)
    np.testing.assert_array_equal(
        returned.filter(like="logit_").to_numpy(), [[5.0, 1.0, -2.0]] * 2
    )
    with pytest.raises(FileExistsError):
        model.predict_dataloader([inputs], tmp_path)
    model.predict_dataloader([batch(("replacement",))], tmp_path, overwrite=True)
    assert list(pd.read_csv(tmp_path / "predictions.csv")["id"]) == ["replacement"]


def test_csv_alone_blocks_overwrite_before_inference(tmp_path):
    (tmp_path / "predictions.csv").write_text("original")
    with pytest.raises(FileExistsError, match="predictions.csv"):
        ensemble(METADATA).predict_dataloader([batch()], tmp_path)
    assert (tmp_path / "predictions.csv").read_text() == "original"
    assert not list(tmp_path.glob("*.npy"))


def test_empty_export_has_readable_columns(tmp_path):
    manifest = ensemble(METADATA).predict_dataloader([], tmp_path)
    assert manifest.empty
    table = pd.read_csv(tmp_path / "predictions.csv")
    assert table.empty and "probability_usable" in table


def test_duplicate_ids_still_fail_when_overwriting(tmp_path):
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        ensemble(METADATA).predict_dataloader(
            [batch(("same",)), batch(("same",))], tmp_path, overwrite=True
        )


def test_runtime_override_reads_names_from_existing_release(tmp_path):
    path = tmp_path / "old.pt"
    old_config = {"inference": {}, "datamodule": {}, "lightningmodule": {"n_class": 3}}
    traced = torch.jit.trace(Members(), torch.zeros(1, 3, 8, 8))
    torch.jit.save(
        traced, str(path), _extra_files={"config.yaml": json.dumps(old_config)}
    )
    before = path.read_bytes()
    model = ClassificationEnsemble.from_torchscript(
        path, inference_overrides={"classification": METADATA}
    )
    inputs = batch()
    result = model.postprocess_batch(model.predict_batch(inputs), inputs)[0]
    assert result["label"] == "good"
    assert path.read_bytes() == before


def test_metadata_normalization_does_not_modify_the_supplied_config():
    metadata = {
        "task": "multilabel",
        "label_map": dict(METADATA["label_map"]),
        "threshold": [0.5, 0.9, 0.1],
    }
    before = deepcopy(metadata)
    model = ensemble(metadata)
    table = model.predict_dataloader([batch()])
    assert metadata == before
    assert model.config["inference"]["classification"] == before
    assert table.loc[0, "labels"] == ["good", "bad"]
    assert "logit_good" in table


def test_actual_channels_are_checked_for_thresholds_without_label_names():
    model = ClassificationEnsemble(
        Members(),
        {
            "datamodule": {},
            "inference": {
                "classification": {"task": "multilabel", "threshold": [0.4, 0.6]}
            },
        },
    )
    inputs = batch()
    with pytest.raises(ValueError, match=r"one value per output channel \(3\)"):
        model.postprocess_batch(model.predict_batch(inputs), inputs)
