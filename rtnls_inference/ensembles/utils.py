import torch
from torch import nn


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
        x = self.ensemble(x)
        x = torch.unbind(x, dim=0)
        return x
