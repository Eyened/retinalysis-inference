from rtnls_inference.ensembles.predict_output import (
    require_tensor,
)

from .base import FundusEnsemble


class EmbeddingEnsemble(FundusEnsemble):
    """Average normalized embeddings across folds."""

    prediction_prefix = "embedding"

    def forward(self, img):
        return self.ensemble(img)

    def _predict_member_tensors(self, batch):
        prediction = self.forward(batch["image"])
        require_tensor(prediction, "prediction", floating=True)
        if prediction.ndim == 2:
            prediction = prediction[None, ...]
        if prediction.ndim != 3:
            raise ValueError(
                f"Embedding backend must return MND, got {prediction.shape}"
            )
        return prediction.permute(1, 0, 2)
