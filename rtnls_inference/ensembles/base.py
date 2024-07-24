import json
import os
from pathlib import Path

import albumentations as A
import lightning as L
import numpy as np
import pandas as pd
from rtnls_fundusprep.colors import contrast_enhance
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from rtnls_inference.datasets.fundus import (
    FundusTestDataset,
)
from rtnls_inference.transforms import make_test_transform
from rtnls_inference.utils import test_collate_fn


class Ensemble(L.LightningModule):
    pass


class FundusEnsemble(Ensemble):
    def __init__(
        self,
        ensemble: L.LightningModule,
        config: dict,
    ):
        super().__init__()
        self.ensemble = ensemble
        self.config = config
        # self.preprocess = preprocess
        # self.transform = make_test_transform(self.config)

        # self.to_tensor = ToTensorV2()
        # self.normalize = A.Compose(
        #     [
        #         A.Normalize(
        #             mean=(0.485, 0.456, 0.406),
        #             std=(0.229, 0.224, 0.225),
        #             max_pixel_value=1,
        #         )
        #     ],
        #     additional_targets={"ce": "image"},
        # )

    def make_batch(self, images):
        batch = []
        for image in images:
            item = {"image": image, "keypoints": []}
            item = self.transform(**item, preprocess=self.preprocess)
            item = self.normalize(**item)
            if "ce" in item:
                item["image"] = np.concatenate([item["image"], item.pop("ce")], axis=-1)
            item = self.to_tensor(**item)
            batch.append(item)

        return test_collate_fn(batch)

    def _make_dataloader(
        self,
        image_paths,
        preprocess,
        batch_size=None,
        num_workers=8,
        ignore_exceptions=True,
    ):
        contrast_enhance=isinstance(image_paths[0], str) or (len(image_paths[0]) == 1)
        dataset = FundusTestDataset(
            images_paths=image_paths,
            transform=make_test_transform(
                self.config,
                preprocess=preprocess,
                contrast_enhance = contrast_enhance,
            ),
        )

        batch_size = (
            batch_size
            if batch_size is not None
            else self.config["inference"].get("batch_size", 8)
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=False,
            shuffle=False,
            collate_fn=(
                test_collate_fn
                if ignore_exceptions
                else torch.utils.data.dataloader.default_collate
            ),
            num_workers=num_workers,
        )

    def predict(
        self,
        image_paths,
        dest_path=None,
        num_workers=0,
        batch_size=None,
    ):
        dataloader = self._make_dataloader(
            image_paths, num_workers=num_workers, preprocess=True, batch_size=batch_size
        )
        return self._predict_dataloader(dataloader, dest_path)

    def predict_preprocessed(
        self,
        image_paths,
        dest_path=None,
        num_workers=0,
        batch_size=None,
    ):
        dataloader = self._make_dataloader(
            image_paths,
            num_workers=num_workers,
            preprocess=False,
            batch_size=batch_size,
        )
        return self._predict_dataloader(dataloader, dest_path)
    
    def predict_dataframe(
        self,
        df: pd.DataFrame,
        dest_path=None,
        image_path_column='image',
        preprocess=True,
        **kwargs
    ):
        image_paths = df[image_path_column].to_list()
        if preprocess:
            return self.predict(image_paths, dest_path, **kwargs)
        else:
            return self.predict_preprocessed(image_paths, dest_path, **kwargs)


    def get_device(self):
        # Check if the module has any parameters
        if next(self.parameters(), None) is not None:
            # Return the device of the first parameter
            return next(self.parameters()).device
        else:
            # Fallback or default device if the module has no parameters
            # This might be necessary for modules that do not have parameters
            # and hence might not have a clear device assignment.
            # Adjust this part based on your specific needs.
            return torch.device("cpu")

    @classmethod
    def from_torchscript(cls, fpath: str | Path):
        extra_files = {"config.yaml": ""}  # values will be replaced with data

        ensemble = torch.jit.load(fpath, _extra_files=extra_files).eval()

        config = json.loads(extra_files["config.yaml"])
        return cls(ensemble, config)

    @classmethod
    def from_release(cls, fname: str):
        if os.path.exists(fname):
            fpath = fname
        else:
            fpath = os.path.join(os.environ["RTNLS_MODEL_RELEASES"], fname)

        fpath = Path(fpath)
        if fpath.suffix == ".pt":
            return cls.from_torchscript(fpath)
        else:
            raise ValueError(f"Unrecognized extension {fpath.suffix}")
        
