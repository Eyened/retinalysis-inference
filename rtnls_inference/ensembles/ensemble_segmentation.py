import os
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from PIL import Image
from tqdm import tqdm

from rtnls_inference.ensembles.utils import EnsembleSplitter

from .base import FundusEnsemble


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def flip(data, axis):
    return torch.flip(data, dims=axis)


class SegmentationEnsemble(FundusEnsemble):
    def forward(self, img):
        """Returns output tensor with shape MNCHW where M=nfolds, the number of models"""
        tta = self.config["inference"].get("tta", False)
        if tta:
            return self.tta_inference(img)
        else:
            return self.sliding_window_inference(img)

    def tta_inference(self, img):
        tta_flips = self.config["inference"].get("tta_flips", [[2], [3], [2, 3]])
        pred = self.sliding_window_inference(img)
        for flip_idx in tta_flips:
            flip_undo_idx = [e + 1 for e in flip_idx]  # output has extra first dim M
            pred += flip(
                self.sliding_window_inference(flip(img, flip_idx)), flip_undo_idx
            )
        pred /= len(tta_flips) + 1
        return pred  # MNCHW

    def sliding_window_inference(self, image):
        patch_size = self.config["inference"].get("tracing_input_size", [512, 512])
        model = EnsembleSplitter(self.ensemble)
        pred = sliding_window_inference(
            inputs=image,
            roi_size=patch_size,
            sw_batch_size=1,
            predictor=model,
            overlap=self.config["inference"].get("overlap", 0.5),
            mode=self.config["inference"].get("blend", "gaussian"),
            device=torch.device("cpu"),
        )
        return torch.stack(pred)  # MNCHW

    def predict_batch(self, batch):
        with torch.autocast(device_type=self.get_device().type):
            proba = self.forward(batch["image"].to(self.get_device()))
        proba = torch.mean(proba, dim=0)  # average over models
        proba = torch.permute(proba, (0, 2, 3, 1))  # NCHW -> NHWC
        proba = torch.nn.functional.softmax(proba, dim=-1)

        # we make a pseudo-batch with the outputs and everything needed for undoing transforms
        items = {
            "id": batch["id"],
            "image": proba,
        }
        return items

    def _save_item(self, item: dict, dest_path: str | Path):
        mask = np.argmax(item["image"], -1)
        Image.fromarray(mask.squeeze().astype(np.uint8)).save(dest_path)

    def _predict_dataloader(self, dataloader, dest_path):
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        with torch.no_grad():
            for batch in tqdm(dataloader):
                if len(batch) == 0:
                    continue

                items = self._predict_batch(batch)

                for i, item in enumerate(items):
                    fpath = os.path.join(dest_path, f'{item["id"]}.png')
                    self._save_item(item, fpath)
