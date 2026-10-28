import torch
from torch import nn

from rtnls_inference.ensembles.predict_output import require_tensor


class EnsembleAverager(nn.Module):
    """Averages the first dimension of the ensemble output, which is the model dimension (nfolds)"""

    def __init__(self, ensemble):
        super().__init__()
        self.ensemble = ensemble

    def forward(self, x):
        return torch.mean(self.ensemble(x), dim=0)


class EnsembleSplitter(nn.Module):
    """Splits the first dimension of the ensemble output, which is the model dimension (nfolds)"""

    def __init__(self, ensemble):
        super().__init__()
        self.ensemble = ensemble

    def forward(self, x):
        prediction = self.ensemble(x)
        require_tensor(prediction, "tile prediction", rank=5, floating=True)
        if prediction.shape[1] != x.shape[0]:
            raise ValueError(
                f"tile prediction batch size must be {x.shape[0]}, got {prediction.shape[1]}"
            )
        return torch.unbind(prediction, dim=0)
