import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .base import FundusEnsemble


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


class RegressionEnsemble(FundusEnsemble):
    def forward(self, img):
        """Returns output tensor with shape MN where M=nfolds, the number of models"""
        return self.ensemble(img).cpu().detach()

    def predict_step(self, batch):
        return self.forward(batch)

    def _predict_dataloader(self, dataloader, dest_path):
        with torch.no_grad():
            batch_ids = []
            batch_preds = []
            for batch in tqdm(dataloader):
                if len(batch) == 0:
                    continue

                preds = (
                    self.forward(batch["image"].to(self.get_device()))
                    .detach()
                    .cpu()
                    .numpy()
                )  # shape: MNC
                batch_ids.extend(batch["id"])
                batch_preds.append(np.mean(preds, axis=0))

        batch_preds = np.concatenate(batch_preds, axis=0)
        return pd.DataFrame(
            batch_preds,
            index=batch_ids,
        )
