import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from rtnls_inference.utils import move_batch_to_device

from .base import FundusEnsemble


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


class RegressionEnsemble(FundusEnsemble):
    def forward(self, batch):
        all_preds = []
        with torch.no_grad():
            batch = move_batch_to_device(batch, self.get_device())
            for model_i, model in enumerate(self.models):
                all_preds.append(model.predict_step(batch).cpu().detach().numpy())

        mean_preds = np.sum(all_preds, axis=0) / len(all_preds)

        return mean_preds

    def predict_step(self, batch):
        return self.forward(batch)

    def predict_images(self, images):
        batch = self.make_batch(images)
        proba = self.predict_step(batch)
        return proba

    def _predict_dataloader(self, dataloader, dest_path):
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        batch_ids = []
        for batch in tqdm(dataloader):
            if len(batch) == 0:
                continue

            preds = self.predict_step(batch)
            batch_ids.extend(batch["id"])
            preds.append(preds)

        preds = np.concatenate(preds, axis=0)
        return pd.DataFrame(
            preds,
            index=batch_ids,
        )
