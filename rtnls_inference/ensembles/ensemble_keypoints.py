from rtnls_inference.ensembles.predict_output import (
    model_input_size,
    require_tensor,
    restore_points_to_preprocessed,
)

from .ensemble_regression import RegressionEnsemble


class KeypointsEnsemble(RegressionEnsemble):
    def _predict_member_tensors(self, batch):
        images = batch["image"]
        keypoints = self.forward(images)
        require_tensor(keypoints, "keypoints", floating=True)
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

    def _prediction_columns(self, count):
        if count % 2:
            raise ValueError("Keypoint outputs must contain xy pairs")
        return [
            coordinate
            for index in range(count // 2)
            for coordinate in (f"x{index}", f"y{index}")
        ]

    def _summary_columns(self, count=0):
        return ["id", *self._prediction_columns(count)]
