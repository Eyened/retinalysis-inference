import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from rtnls_inference.ensembles.predict_output import decollate_predict_full

from .ensemble_regression import RegressionEnsemble


class ClassificationEnsemble(RegressionEnsemble):
    def _compatibility_items(self, full_output):
        return [
            {"id": item.get("id"), "logits": item["aggregate"]}
            for item in decollate_predict_full(full_output)
        ]

    def _predict_dataloader(self, dataloader, dest_path):
        with torch.no_grad():
            batch_ids = []
            batch_preds = []
            for batch in tqdm(dataloader):
                if len(batch) == 0:
                    continue

                batch_items = self._compatibility_items(self.predict_step_full(batch))
                if not batch_items:
                    continue
                batch_ids.extend(item["id"] for item in batch_items)
                batch_preds.append(
                    np.stack([item["logits"] for item in batch_items], axis=0)
                )

        if not batch_preds:
            return pd.DataFrame(index=batch_ids)
        batch_preds = np.concatenate(batch_preds, axis=0)
        return pd.DataFrame(
            batch_preds,
            index=batch_ids,
        )
