import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from rtnls_inference.ensembles.base import FundusEnsemble
from rtnls_inference.ensembles.utils import EnsembleSplitter
from rtnls_inference.utils import decollate_batch, extract_keypoints_from_heatmaps

from .base import FundusEnsemble


def flip(data, axis):
    return torch.flip(data, dims=axis)


class HeatmapRegressionEnsemble(FundusEnsemble):
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

    def _predict_dataloader(self, dataloader, dest_path=None):
        with torch.no_grad():
            all_kps = []
            all_ids = []
            for batch in tqdm(dataloader):
                if len(batch) == 0:
                    continue

                with torch.autocast(device_type=self.get_device().type):
                    heatmap = self.forward(batch["image"].to(self.get_device()))
                keypoints = extract_keypoints_from_heatmaps(heatmap)

                keypoints = torch.mean(keypoints, dim=1)  # average over models
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
