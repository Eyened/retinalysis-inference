import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from rtnls_inference.utils import decollate_batch

from .ensemble_regression import RegressionEnsemble


class ClassificationEnsemble(RegressionEnsemble):
    def _predict_batch(self, batch: dict) -> list[dict]:
        """Run classification inference for a batch and return decollated outputs."""
        logits = self.forward(batch["image"].to(self.get_device()))  # shape: MNC
        logits = torch.mean(logits, dim=0)  # match inference.py: average logits only
        items = {
            "id": batch["id"],
            "logits": logits,
        }
        return decollate_batch(items)

    def _predict_dataloader(self, dataloader, dest_path):
        with torch.no_grad():
            batch_ids = []
            batch_preds = []
            for batch in tqdm(dataloader):
                if len(batch) == 0:
                    continue

                batch_items = self._predict_batch(batch)
                if not batch_items:
                    continue
                batch_ids.extend(item["id"] for item in batch_items)
                batch_preds.append(
                    np.stack([item["logits"] for item in batch_items], axis=0)
                )

        batch_preds = np.concatenate(batch_preds, axis=0)
        return pd.DataFrame(
            batch_preds,
            index=batch_ids,
        )
