import os
import numpy as np
import pandas as pd
from rtnls_inference.utils import decollate_batch
import torch
from tqdm import tqdm

from .ensemble_regression import RegressionEnsemble


class KeypointsEnsemble(RegressionEnsemble):
    def _predict_dataloader(self, dataloader, dest_path=None):
        with torch.no_grad():
            all_kps = []
            all_ids = []

            for batch in tqdm(dataloader):
                if len(batch) == 0:
                    continue

                keypoints = self.forward(batch["image"].to(self.get_device()))

                keypoints = keypoints[:,:,None,:] # MNC2 (models, batch_size, num_keypoints, 2)

                if self.config['datamodule'].get('normalize_keypoints', True):
                    n, c, h, w = batch["image"].shape
                    keypoints[..., 0] = (keypoints[..., 0]) * w
                    keypoints[..., 1] = (keypoints[..., 1]) * h

                keypoints = torch.mean(keypoints, dim=0)  # average over models
                # we make a pseudo-batch with the outputs and everything needed for undoing transforms

                items = {
                    "id": batch["id"],
                    "keypoints": keypoints,
                }
                if "bounds" in batch:
                    items["bounds"] = batch["bounds"]
                items = decollate_batch(items)

                items = [dataloader.dataset.transform.undo_item(item) for item in items]
                all_ids += [item["id"] for item in items]
                all_kps += [item["keypoints"] for item in items]


            columns = [(f"x{i}", f"y{i}") for i in range(len(all_kps[0]))]
            columns = [item for sublist in columns for item in sublist]
            all_kps = [kp.flatten() for kp in all_kps]
            return pd.DataFrame(all_kps, index=all_ids, columns=columns)
