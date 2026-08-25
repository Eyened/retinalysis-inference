import numpy as np
import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from pydantic import Field, model_validator
from tqdm import tqdm

from rtnls_inference.ensembles.ensemble_keypoints import (
    KeypointsEnsemble,
    KeypointsPredictFull,
)
from rtnls_inference.ensembles.predict_output import (
    decollate_predict_full,
)
from rtnls_inference.ensembles.utils import EnsembleSplitter
from rtnls_inference.utils import extract_keypoints_from_heatmaps


def flip(data, axis):
    return torch.flip(data, dims=axis)


class HeatmapRegressionPredictFull(KeypointsPredictFull):
    """Keypoint predictions with optional per-member heatmaps."""

    heatmaps: np.ndarray | None = Field(
        default=None,
        description="Optional per-member heatmaps as NMKHW.",
    )

    @model_validator(mode="after")
    def validate_heatmaps(self):
        if self.heatmaps is not None:
            if self.heatmaps.ndim != 5:
                raise ValueError("heatmaps must have shape NMKHW")
            if self.heatmaps.shape[:3] != self.prediction.shape[:3]:
                raise ValueError(
                    "heatmap and keypoint batch/member/keypoint axes differ"
                )
        return self


class HeatmapRegressionEnsemble(KeypointsEnsemble):
    predict_full_model = HeatmapRegressionPredictFull
    autocast_inference = True

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
        )
        if isinstance(pred, tuple):
            pred = torch.stack(pred, dim=1)

        if pred.dim() == 4:
            pred = pred[:, None, ...]

        return pred  # NMCHW

    def _predict_member_tensors(self, batch):
        heatmaps = self.forward(batch["image"])
        return extract_keypoints_from_heatmaps(heatmaps)

    def _predict_step_tensors(self, batch):
        heatmaps = self.forward(batch["image"])
        prediction = extract_keypoints_from_heatmaps(heatmaps)
        return {
            "prediction": prediction,
            "aggregate": self._aggregate_tensors(prediction),
            "heatmaps": heatmaps,
        }

    def _prediction_size(self, batch, output):
        heatmaps = output["heatmaps"]
        return int(heatmaps.shape[-2]), int(heatmaps.shape[-1])

    def _predict_dataloader(self, dataloader, dest_path=None):
        with torch.no_grad():
            all_kps = []
            all_ids = []
            for batch in tqdm(dataloader):
                if len(batch) == 0:
                    continue

                items = [
                    self.postprocess_item(item)
                    for item in decollate_predict_full(self.predict_step_full(batch))
                ]
                all_ids += [item["id"] for item in items]
                all_kps += [item["keypoints"] for item in items]

            if not all_kps:
                return pd.DataFrame(index=all_ids)
            columns = [(f"x{i}", f"y{i}") for i in range(len(all_kps[0]))]
            columns = [item for sublist in columns for item in sublist]
            all_kps = [kp.flatten() for kp in all_kps]
            return pd.DataFrame(all_kps, index=all_ids, columns=columns)
