import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .ensemble_regression import RegressionEnsemble


class ClassificationEnsemble(RegressionEnsemble):
    def _predict_dataloader(self, dataloader, dest_path):
        with torch.no_grad():
            batch_ids = []
            batch_preds = []
            for batch in tqdm(dataloader):
                if len(batch) == 0:
                    continue

                logits = self.forward(
                    batch["image"].to(self.get_device())
                )  # shape: MNC

                proba = torch.nn.functional.softmax(logits, dim=-1)
                proba = torch.mean(proba, dim=0)  # average over models

                batch_ids.extend(batch["id"])
                batch_preds.append(proba.numpy())

        batch_preds = np.concatenate(batch_preds, axis=0)
        return pd.DataFrame(
            batch_preds,
            index=batch_ids,
        )
