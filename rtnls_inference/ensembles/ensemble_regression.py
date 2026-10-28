import numpy as np

from rtnls_inference.ensembles.predict_output import (
    require_tensor,
)

from .base import FundusEnsemble


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


class RegressionEnsemble(FundusEnsemble):
    def forward(self, img):
        """Return backend member predictions, normally MNC."""
        return self.ensemble(img)

    def _predict_member_tensors(self, batch):
        prediction = self.forward(batch["image"])
        require_tensor(prediction, "prediction", floating=True)
        if prediction.ndim == 2:
            prediction = prediction[..., None]
        if prediction.ndim != 3:
            raise ValueError(
                f"Regression backend must return MNC, got {prediction.shape}"
            )
        return prediction.permute(1, 0, 2)
