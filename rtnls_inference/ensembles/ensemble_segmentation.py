import numpy as np
import pandas as pd
import torch
from monai.inferers import sliding_window_inference

from rtnls_inference.ensembles.output_files import (
    MANIFEST_COLUMNS,
    output_suffix,
    prepare_export,
    save_output_item,
)
from rtnls_inference.ensembles.predict_output import (
    output_manifest_row,
    require_tensor,
    restore_array_to_preprocessed,
)
from rtnls_inference.ensembles.utils import EnsembleSplitter
from rtnls_inference.inputs import validate_ids
from rtnls_inference.metrics import Dice

from .base import FundusEnsemble


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def flip(data, axis):
    return torch.flip(data, dims=axis)


class SegmentationEnsemble(FundusEnsemble):
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
        prediction = self.forward(batch["image"])
        require_tensor(prediction, "segmentation prediction", rank=5, floating=True)
        return prediction.permute(0, 1, 3, 4, 2)

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
        require_tensor(aggregate, "aggregate", rank=4, floating=True)
        if aggregate.shape != logits.shape:
            raise ValueError(
                f"prediction and aggregate output shapes differ: expected {tuple(logits.shape)}, got {tuple(aggregate.shape)}"
            )
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

    def predict_dataloader(self, dataloader, dest_path=None, *, overwrite=False):
        """Write final masks incrementally and return their manifest.

        Use iter_predictions for in-memory consumption of spatial outputs.
        """
        if dest_path is None:
            raise ValueError(
                "dest_path is required for segmentation; use iter_predictions for in-memory results"
            )
        manifest_path = prepare_export(
            dataloader, dest_path, "manifest.csv", overwrite=overwrite
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for item in self.iter_predictions(dataloader):
            identifier = validate_ids([item.get("id")], for_output=True)[0]
            path = manifest_path.parent / f"{identifier}{output_suffix(item)}"
            if path.exists() and not overwrite:
                raise FileExistsError(path)
            save_output_item(item, path)
            rows.append(output_manifest_row(item, str(path)))
        manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
        manifest.to_csv(manifest_path, index=False)
        return manifest

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
