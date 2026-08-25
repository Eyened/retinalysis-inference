import pandas as pd
import torch
from pydantic import model_validator
from tqdm import tqdm

from rtnls_inference.ensembles.predict_output import (
    PredictFullOutput,
    decollate_predict_full,
    model_input_size,
    require_rank,
    restore_points_to_preprocessed,
)

from .ensemble_regression import RegressionEnsemble


class KeypointsPredictFull(PredictFullOutput):
    @model_validator(mode="after")
    def validate_keypoints(self):
        require_rank(self.prediction, 4, "prediction")
        require_rank(self.aggregate, 3, "aggregate")
        if self.prediction.shape[-1] != 2 or self.aggregate.shape[-1] != 2:
            raise ValueError("keypoints must have a final xy axis")
        if self.prediction.shape[2:] != self.aggregate.shape[1:]:
            raise ValueError("member and aggregate keypoint shapes differ")
        return self


class KeypointsEnsemble(RegressionEnsemble):
    predict_full_model = KeypointsPredictFull

    def _predict_member_tensors(self, batch):
        images = batch["image"]
        keypoints = self.forward(images)
        if keypoints.ndim == 2:
            keypoints = keypoints[None, ...]
        if keypoints.ndim == 3:
            if keypoints.shape[-1] % 2:
                raise ValueError("flattened keypoint output must contain xy pairs")
            keypoints = keypoints.reshape(keypoints.shape[0], keypoints.shape[1], -1, 2)
        if keypoints.ndim != 4 or keypoints.shape[-1] != 2:
            raise ValueError(
                f"Keypoint backend must return MNK2, got {keypoints.shape}"
            )

        if self.config["datamodule"].get("normalize_keypoints", True):
            _, _, h, w = images.shape
            keypoints[..., 0] *= w
            keypoints[..., 1] *= h
        return keypoints.permute(1, 0, 2, 3)

    def _prediction_size(self, batch, output):
        return model_input_size(batch)

    def postprocess_item(self, item):
        keypoints = restore_points_to_preprocessed(item["aggregate"], item["geometry"])
        result = {
            "id": item.get("id"),
            "output": keypoints,
            "keypoints": keypoints,
            "output_kind": "keypoints",
            "output_space": "preprocessed",
            "geometry": item["geometry"],
        }
        if "preprocessed_image" in item:
            result["preprocessed_image"] = item["preprocessed_image"]
        return result

    def _compatibility_items(self, full_output):
        return [
            {"id": item.get("id"), "keypoints": item["aggregate"]}
            for item in decollate_predict_full(full_output)
        ]

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
