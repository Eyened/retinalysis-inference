import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from rtnls_inference.utils import decollate_batch

from .base import FundusEnsemble


class EmbeddingEnsemble(FundusEnsemble):
    """Average normalized embeddings across folds."""

    def forward(self, img):
        return self.ensemble(img).cpu().detach()

    def _predict_batch(self, batch: dict) -> list[dict]:
        images = batch["image"].to(self.get_device())
        embeddings = torch.mean(self.forward(images), dim=0)
        items = {
            "id": batch["id"],
            "embedding": embeddings,
        }
        return decollate_batch(items)

    def _predict_dataloader(self, dataloader, dest_path):
        with torch.no_grad():
            batch_ids = []
            batch_embeddings = []
            for batch in tqdm(dataloader):
                if len(batch) == 0:
                    continue

                batch_items = self._predict_batch(batch)
                if not batch_items:
                    continue
                batch_ids.extend(item["id"] for item in batch_items)
                batch_embeddings.append(
                    np.stack([item["embedding"] for item in batch_items], axis=0)
                )

        batch_embeddings = np.concatenate(batch_embeddings, axis=0)
        return pd.DataFrame(batch_embeddings, index=batch_ids)

