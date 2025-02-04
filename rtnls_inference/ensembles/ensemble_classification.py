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

                logits = torch.mean(logits, dim=0)
                # proba = torch.nn.functional.softmax(torch.mean(logits, dim=0), dim=-1)

                batch_ids.extend(batch["id"])
                batch_preds.append(logits.numpy())

        batch_preds = np.concatenate(batch_preds, axis=0)
        return pd.DataFrame(
            batch_preds,
            index=batch_ids,
        )
