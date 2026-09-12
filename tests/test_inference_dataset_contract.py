from pydantic import ValidationError
import pytest

from rtnls_inference.datasets.fundus import FundusInferenceDataset
from rtnls_inference.dtos_inference import InferenceDatasetDTO, ModelInputDTO


def test_inference_dataset_owns_strict_model_input_contract():
    assert FundusInferenceDataset.item_dto_cls is ModelInputDTO
    assert FundusInferenceDataset.dataset_dto_cls == InferenceDatasetDTO[ModelInputDTO]


@pytest.mark.parametrize("field", ["mask", "labels", "keypoints", "logits"])
def test_inference_contract_rejects_evaluation_and_training_targets(field):
    with pytest.raises(ValidationError):
        FundusInferenceDataset({"images": [{"id": "sample", field: "value"}]})
