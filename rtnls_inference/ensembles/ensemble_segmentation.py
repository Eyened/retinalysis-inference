import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from PIL import Image
from pydantic import Field, model_validator
from tqdm import tqdm

from rtnls_inference.ensembles.predict_output import (
    PredictFullOutput,
    decollate_predict_full,
    output_manifest_row,
    require_rank,
    restore_array_to_preprocessed,
)
from rtnls_inference.ensembles.utils import EnsembleSplitter
from rtnls_inference.metrics import Dice

from .base import FundusEnsemble


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def flip(data, axis):
    return torch.flip(data, dims=axis)


class SegmentationPredictFull(PredictFullOutput):
    logits: np.ndarray = Field(
        description="Ensemble-averaged pre-softmax logits as NHWC."
    )

    @model_validator(mode="after")
    def validate_segmentation(self):
        require_rank(self.prediction, 5, "prediction")
        require_rank(self.aggregate, 4, "aggregate")
        require_rank(self.logits, 4, "logits")
        if self.prediction.shape[2:] != self.logits.shape[1:]:
            raise ValueError("member logits and averaged logits shapes differ")
        if self.aggregate.shape != self.logits.shape:
            raise ValueError("aggregate probabilities and logits shapes differ")
        if not all(
            np.issubdtype(value.dtype, np.floating)
            for value in (self.prediction, self.aggregate, self.logits)
        ):
            raise ValueError(
                "segmentation predictions and logits must be floating point"
            )
        return self


class SegmentationEnsemble(FundusEnsemble):
    predict_full_model = SegmentationPredictFull
    autocast_inference = True

    def __init__(self, *args, postprocess_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.postprocess_fn = postprocess_fn

    def forward(self, img):
        """Returns output tensor with shape MNCHW where M=nfolds, the number of models"""
        if self._inference_setting("tta", False):
            return self.tta_inference(img)
        return self.sliding_window_inference(img)

    def _predict_member_tensors(self, batch):
        return self.forward(batch["image"]).permute(0, 1, 3, 4, 2)

    def _aggregate_tensors(self, member_output):
        return torch.softmax(member_output.mean(dim=1), dim=-1)

    def _predict_step_tensors(self, batch):
        prediction = self._predict_member_tensors(batch)
        expected_channels = self.config.get("lightningmodule", {}).get("n_class")
        if expected_channels is not None and prediction.shape[-1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} segmentation channels, "
                f"got {prediction.shape[-1]}"
            )
        logits = prediction.mean(dim=1)
        aggregate = self._aggregate_tensors(prediction)
        return {"prediction": prediction, "logits": logits, "aggregate": aggregate}

    def _prediction_size(self, batch, output):
        aggregate = output["aggregate"]
        return int(aggregate.shape[1]), int(aggregate.shape[2])

    def tta_inference(self, img):
        tta_flips = self._inference_setting("tta_flips", [[2], [3], [2, 3]])
        pred = self.sliding_window_inference(img)
        for flip_idx in tta_flips:
            flip_undo_idx = [e + 1 for e in flip_idx]  # output has extra first dim M
            pred += flip(
                self.sliding_window_inference(flip(img, flip_idx)), flip_undo_idx
            )
        pred /= len(tta_flips) + 1
        return pred  # MNCHW

    def sliding_window_inference(self, image):
        patch_size = self._inference_setting("tracing_input_size", [512, 512])
        model = EnsembleSplitter(self.ensemble)
        pred = sliding_window_inference(
            inputs=image,
            roi_size=patch_size,
            sw_batch_size=self._tile_batch_size(16),
            predictor=model,
            overlap=self._inference_setting("overlap", 0.5),
            mode=self._inference_setting("blend", "gaussian"),
            # device=torch.device("cpu"),
        )
        if isinstance(pred, tuple):
            pred = torch.stack(pred, dim=1)

        if pred.dim() == 4:
            pred = pred[:, None, ...]

        return pred  # NMCHW

    def postprocess_item(self, item):
        probabilities = restore_array_to_preprocessed(
            item["aggregate"], item["geometry"], "bilinear"
        )
        mask = np.argmax(probabilities, axis=-1).astype(np.uint8)
        if self.postprocess_fn is not None:
            mask = self.postprocess_fn(mask)
        result = {
            "id": item.get("id"),
            "output": mask,
            "probabilities": probabilities,
            "output_kind": "mask",
            "output_space": "preprocessed",
            "geometry": item["geometry"],
        }
        if "preprocessed_image" in item:
            result["preprocessed_image"] = item["preprocessed_image"]
        return result

    @staticmethod
    def _save_item(item: dict, dest_path: str | Path):
        Image.fromarray(np.asarray(item["output"], dtype=np.uint8)).save(dest_path)

    def _compatibility_items(self, full_output):
        return [
            {"id": item.get("id"), "image": item["aggregate"]}
            for item in decollate_predict_full(full_output)
        ]

    def _predict_output_batch(self, batch: dict) -> list[dict]:
        return [
            self.postprocess_item(item)
            for item in decollate_predict_full(self.predict_step_full(batch))
        ]

    def _predict_dataloader(
        self,
        dataloader,
        dest_path,
    ):
        if dest_path is None:
            raise ValueError("dest_path is required for spatial prediction output")
        os.makedirs(dest_path, exist_ok=True)
        manifest = []
        for batch in tqdm(dataloader):
            if len(batch) == 0:
                continue
            for item in self._predict_output_batch(batch):
                fpath = Path(dest_path) / f"{item['id']}.png"
                self._save_item(item, fpath)
                manifest.append(output_manifest_row(item, str(fpath)))
        return pd.DataFrame(manifest)

    def on_test_start(self):
        self.dice = Dice(self.config["lightningmodule"].get("n_class", 2))

    def test_step(self, batch, batch_idx):
        proba = self.forward(batch["image"])
        proba = torch.mean(proba, dim=1)  # average over models, NCHW

        mask = batch["mask"][..., 1][:, None, :, :] == 0
        lbl = batch["mask"][..., 0][:, None, :, :]

        # shapes: BNHWD
        metrics, _ = self.dice(proba, lbl[:, 0], mask[:, 0], 0)
        self.log_dict(
            {f"C{i}": v for i, v in enumerate(metrics)}, on_step=False, on_epoch=True
        )
