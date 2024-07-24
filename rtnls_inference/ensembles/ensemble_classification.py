import numpy as np
import pandas as pd
from tqdm import tqdm

from .ensemble_regression import RegressionEnsemble


class ClassificationEnsemble(RegressionEnsemble):
    def _predict_dataloader(self, dataloader, dest_path):
        ids = []
        preds = []
        for batch in tqdm(dataloader):
            if len(batch) == 0:
                continue

            batch_preds = self.predict_step(batch)
            batch_preds = dataloader.dataset.transform.undo_keypoints(
                batch, batch_preds
            )
            ids.extend(batch["id"])
            preds.append(batch_preds)

        preds = np.concatenate(preds, axis=0)
        return pd.DataFrame(
            preds,
            index=ids,
        )
