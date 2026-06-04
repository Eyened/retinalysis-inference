import os
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from PIL import Image
from pytorch_lightning.utilities import move_data_to_device
from tqdm import tqdm

from rtnls_inference.ensembles.utils import EnsembleSplitter
from rtnls_inference.metrics import Dice
from rtnls_inference.utils import decollate_batch

from .base import FundusEnsemble


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def flip(data, axis):
    return torch.flip(data, dims=axis)


class SegmentationEnsemble(FundusEnsemble):
    def __init__(self, *args, postprocess_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.postprocess_fn = postprocess_fn

    def forward(self, img):
        """Returns output tensor with shape MNCHW where M=nfolds, the number of models"""
        tta = self.config["inference"].get("tta", False)
        if tta:
            return self.tta_inference(img)
        else:
            return self.sliding_window_inference(img)

    def predict_step(self, batch, batch_idx=None):
        """Returns the output averaged over models, shape NHWC"""
        logits = self.predict_logits_step(batch)
        return torch.nn.functional.softmax(logits, dim=-1)

    def predict_logits_step(self, batch, batch_idx=None):
        """Returns ensemble-averaged pre-softmax logits, shape NHWC."""
        return self._predict_logits_tensor(batch)

    def _predict_logits_tensor(self, batch):
        """Return ensemble-averaged pre-softmax logits as an NHWC tensor."""
        logits = self.forward(batch["image"])
        logits = torch.mean(logits, dim=1)  # average over models
        logits = torch.permute(logits, (0, 2, 3, 1))  # NCHW -> NHWC
        return logits

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
            sw_batch_size=16,
            predictor=model,
            overlap=self.config["inference"].get("overlap", 0.5),
            mode=self.config["inference"].get("blend", "gaussian"),
            # device=torch.device("cpu"),
        )
        if isinstance(pred, tuple):
            pred = torch.stack(pred, dim=1)

        if pred.dim() == 4:
            pred = pred[:, None, ...]

        return pred  # NMCHW

    def _save_item(self, item: dict, dest_path: str | Path):
        mask = np.argmax(item["image"], -1)
        mask = mask.squeeze().astype(np.uint8)
        if self.postprocess_fn is not None:
            mask = self.postprocess_fn(mask)

        Image.fromarray(mask).save(dest_path)

    def _save_logits_item(
        self,
        item: dict,
        dest_path: str | Path,
        dtype: np.dtype | type = np.float16,
    ):
        logits = np.asarray(item["image"], dtype=dtype)
        np.save(dest_path, logits)

    def _predict_batch(self, batch: dict) -> list[dict]:
        """Run segmentation inference for a batch and return decollated outputs."""
        return self._predict_output_batch(batch, output="proba")

    def _predict_logits_batch(self, batch: dict) -> list[dict]:
        """Run segmentation inference for a batch and return raw logit maps."""
        return self._predict_output_batch(batch, output="logits")

    def _predict_output_batch(self, batch: dict, output: str = "proba") -> list[dict]:
        """Run segmentation inference for a batch and return decollated outputs."""
        with torch.autocast(device_type=self.get_device().type):
            batch_on_device = move_data_to_device(batch, self.get_device())
            if output == "proba":
                image = self.predict_step(batch_on_device)
            elif output == "logits":
                image = self.predict_logits_step(batch_on_device)
            else:
                raise ValueError(f"Invalid segmentation output: {output}")

        items = {
            "id": batch["id"],
            "image": image,
        }
        if "bounds" in batch:
            items["bounds"] = batch["bounds"]
        if "metadata" in batch:
            items["metadata"] = batch["metadata"]
        return decollate_batch(items)

    def _predict_dataloader(
        self,
        dataloader,
        dest_path,
        output: str = "proba",
        suffix: str = ".png",
        dtype: np.dtype | type = np.float16,
    ):
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        with torch.no_grad():
            for batch in tqdm(dataloader):
                if len(batch) == 0:
                    continue

                items = self._predict_output_batch(batch, output=output)
                items = [dataloader.dataset.transform.undo_item(item) for item in items]

                for item in items:
                    fpath = os.path.join(dest_path, f"{item['id']}{suffix}")
                    if output == "logits":
                        self._save_logits_item(item, fpath, dtype=dtype)
                    else:
                        self._save_item(item, fpath)

    def _predict_logits_dataloader(
        self,
        dataloader,
        dest_path,
        dtype: np.dtype | type = np.float16,
    ):
        return self._predict_dataloader(
            dataloader,
            dest_path,
            output="logits",
            suffix=".npy",
            dtype=dtype,
        )

    def predict_logits_dataset(
        self,
        data,
        dest_path,
        num_workers=0,
        batch_size=None,
        dtype: np.dtype | type = np.float16,
    ):
        """Run inference on a dataset and save pre-softmax logits as .npy files."""
        inputs = {"images": data}
        dataloader = self._make_inference_dataloader(
            inputs,
            num_workers=num_workers,
            preprocess=True,
            batch_size=batch_size,
        )
        return self._predict_logits_dataloader(dataloader, dest_path, dtype=dtype)

    def predict_logits_preprocessed(
        self,
        data,
        dest_path,
        num_workers=0,
        batch_size=None,
        dtype: np.dtype | type = np.float16,
    ):
        """Run inference on preprocessed images and save pre-softmax logits."""
        inputs = {"images": data}
        dataloader = self._make_inference_dataloader(
            inputs,
            num_workers=num_workers,
            preprocess=False,
            batch_size=batch_size,
        )
        return self._predict_logits_dataloader(dataloader, dest_path, dtype=dtype)

    def on_test_start(self):
        self.dice = Dice(self.config["lightningmodule"].get("n_class", 2))

    def test_step(self, batch, batch_idx):
        proba = self.forward(batch["image"])
        proba = torch.mean(proba, dim=0)  # average over models, NCHW

        mask = batch["mask"][..., 1][:, None, :, :] == 0
        lbl = batch["mask"][..., 0][:, None, :, :]

        # shapes: BNHWD
        metrics, _ = self.dice(proba, lbl[:, 0], mask[:, 0], 0)
        self.log_dict(
            {f"C{i}": v for i, v in enumerate(metrics)}, on_step=False, on_epoch=True
        )
