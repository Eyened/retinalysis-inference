import numpy as np
import pandas as pd
import torch
from pydantic import model_validator
from tqdm import tqdm

from rtnls_inference.ensembles.predict_output import (
    PredictFullOutput,
    decollate_predict_full,
    require_rank,
)

from .base import FundusEnsemble


class EmbeddingPredictFull(PredictFullOutput):
    @model_validator(mode="after")
    def validate_embedding(self):
        require_rank(self.prediction, 3, "prediction")
        require_rank(self.aggregate, 2, "aggregate")
        if self.prediction.shape[2] != self.aggregate.shape[1]:
            raise ValueError("member and aggregate embedding dimensions differ")
        return self


class EmbeddingEnsemble(FundusEnsemble):
    """Average normalized embeddings across folds."""

    predict_full_model = EmbeddingPredictFull

    def forward(self, img):
        return self.ensemble(img)

    def _predict_member_tensors(self, batch):
        prediction = self.forward(batch["image"])
        if prediction.ndim == 2:
            prediction = prediction[None, ...]
        if prediction.ndim != 3:
            raise ValueError(
                f"Embedding backend must return MND, got {prediction.shape}"
            )
        return prediction.permute(1, 0, 2)

    def _compatibility_items(self, full_output):
        return [
            {"id": item.get("id"), "embedding": item["aggregate"]}
            for item in decollate_predict_full(full_output)
        ]

    def _predict_dataloader(self, dataloader, dest_path):
        with torch.no_grad():
            batch_ids = []
            batch_embeddings = []
            for batch in tqdm(dataloader):
                if len(batch) == 0:
                    continue

                batch_items = self._compatibility_items(self.predict_step_full(batch))
                if not batch_items:
                    continue
                batch_ids.extend(item["id"] for item in batch_items)
                batch_embeddings.append(
                    np.stack([item["embedding"] for item in batch_items], axis=0)
                )

        if not batch_embeddings:
            return pd.DataFrame(index=batch_ids)
        batch_embeddings = np.concatenate(batch_embeddings, axis=0)
        return pd.DataFrame(batch_embeddings, index=batch_ids)
