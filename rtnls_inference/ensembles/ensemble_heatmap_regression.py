import torch
from monai.inferers import sliding_window_inference

from rtnls_inference.ensembles.ensemble_keypoints import (
    KeypointsEnsemble,
)
from rtnls_inference.ensembles.predict_output import require_tensor
from rtnls_inference.ensembles.utils import EnsembleSplitter
from rtnls_inference.utils import extract_keypoints_from_heatmaps


def flip(data, axis):
    return torch.flip(data, dims=axis)


class HeatmapRegressionEnsemble(KeypointsEnsemble):
    autocast_inference = True

    def forward(self, img):
        """Returns output tensor with shape MNCHW where M=nfolds, the number of models"""
        if self._inference_setting("tta", False):
            return self.tta_inference(img)
        return self.sliding_window_inference(img)

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
            sw_batch_size=self._tile_batch_size(1),
            predictor=model,
            overlap=self._inference_setting("overlap", 0.5),
            mode=self._inference_setting("blend", "gaussian"),
        )
        if isinstance(pred, tuple):
            pred = torch.stack(pred, dim=1)

        if pred.dim() == 4:
            pred = pred[:, None, ...]

        return pred  # NMCHW

    def _predict_member_tensors(self, batch):
        heatmaps = self.forward(batch["image"])
        require_tensor(heatmaps, "heatmaps", rank=5)
        return extract_keypoints_from_heatmaps(heatmaps)

    def _predict_step_tensors(self, batch):
        heatmaps = self.forward(batch["image"])
        require_tensor(heatmaps, "heatmaps", rank=5)
        prediction = extract_keypoints_from_heatmaps(heatmaps)
        return {
            "prediction": prediction,
            "aggregate": self._aggregate_tensors(prediction),
            "heatmaps": heatmaps,
        }

    def _prediction_size(self, batch, output):
        heatmaps = output["heatmaps"]
        return int(heatmaps.shape[-2]), int(heatmaps.shape[-1])

    def _summary_columns(self, count=0):
        count = count or 2 * self.config.get("lightningmodule", {}).get("n_class", 0)
        return super()._summary_columns(count)
