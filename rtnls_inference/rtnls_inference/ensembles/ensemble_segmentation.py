import os
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from PIL import Image
from tqdm import tqdm

from rtnls_inference.utils import decollate_batch

from .base import FundusEnsemble


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def flip(data, axis):
    return torch.flip(data, dims=axis)


class SegmentationEnsemble(FundusEnsemble):
    tta_flips = [[2], [3], [2, 3]]

    def forward(self, img):
        tta = self.config["inference"].get("tta", False)
        if tta:
            return self.tta_inference(img)
        else:
            return self.sliding_window_inference(img)

    def tta_inference(self, img):
        pred = self.sliding_window_inference(img)
        for flip_idx in self.tta_flips:
            pred += flip(self.sliding_window_inference(flip(img, flip_idx)), flip_idx)
        pred /= len(self.tta_flips) + 1
        return pred

    def sliding_window_inference(self, image):
        print("here")
        patch_size = self.config["inference"].get("tracing_input_size", [512, 512])
        pred = sliding_window_inference(
            inputs=image,
            roi_size=patch_size,
            sw_batch_size=1,
            predictor=self.ensemble,
            overlap=self.config["inference"].get("overlap", 0.5),
            mode=self.config["inference"].get("blend", "gaussian"),
            device=torch.device("cpu"),
        )
        return pred

    # def predict_step(self, batch):
    #     with torch.no_grad():
    #         img = batch["image"].to(self.get_device())
    #         logits = self.forward(batch)
    #     logits = torch.permute(logits, (0, 2, 3, 1))
    #     proba = torch.nn.functional.softmax(logits, dim=-1)
    #     return proba.cpu().detach().numpy()

    def predict_images(self, images):
        batch = self.make_batch(images)
        proba = self.forward(batch)
        proba = self.transform.undo(batch, proba, preprocess=self.preprocess)
        return proba

    def _save_item(self, item: dict, dest_path: str | Path):
        mask = np.argmax(item["image"], -1)
        id = item["id"]
        Image.fromarray(mask.squeeze().astype(np.uint8)).save(dest_path)

    def _predict_dataloader(self, dataloader, dest_path):
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        with torch.no_grad():
            for batch in tqdm(dataloader):
                if len(batch) == 0:
                    continue

                with torch.autocast(device_type=self.get_device().type):
                    proba = self.forward(batch["image"].to(self.get_device()))
                proba = torch.permute(proba, (0, 2, 3, 1))
                proba = torch.nn.functional.softmax(proba, dim=-1)

                # we make a pseudo-batch with the outputs and everything needed for undoing transforms
                items = {
                    "id": batch["id"],
                    "image": proba,
                }
                if "bounds" in items:
                    items["bounds"] = batch["bounds"]
                items = decollate_batch(items)

                for i, item in enumerate(items):
                    item = dataloader.dataset.transform.undo_item(item)
                    fpath = os.path.join(dest_path, f'{item["id"]}.png')
                    self._save_item(item, fpath)
