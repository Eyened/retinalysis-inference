import warnings
from math import floor, log10
from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

from rtnls_inference.utils import load_image

from .base import TestDataset

normalizer = A.Compose(
    [
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1
        )
    ],
    additional_targets={"ce": "image"},
)

to_tensor = ToTensorV2()


class FundusTestDataset(TestDataset):
    def __init__(
        self,
        images_paths,
        transform=None,
        normalize=True,
        ignore_exceptions=False,
        **kwargs,
    ):
        self.image_paths = images_paths
        self.transform = transform
        self.normalize = normalizer if normalize else lambda **x: x

        self.ignore_exceptions = ignore_exceptions

    def __len__(self):
        return len(self.image_paths)

    def _open_image(self, idx):
        fp = self.image_paths[idx]
        if isinstance(fp, list) or isinstance(fp, tuple):
            if len(fp) == 1:
                return load_image(fp[0], np.float32), None
            elif len(fp) == 2:
                return load_image(fp[0], np.float32), load_image(fp[1], np.float32)
            else:
                raise ValueError(f"Invalid image paths {fp}")
        else:
            return load_image(fp, np.float32), None

    def get_id(self, idx):
        if self.image_paths is not None:
            fp = self.image_paths[idx]
            if isinstance(fp, list) or isinstance(fp, tuple):
                return Path(fp[0]).stem
            else:
                return Path(fp).stem

        else:
            return str(idx).zfill(floor(log10(len(self))))

    def getitem(self, idx, normalize=False):
        image, ce = self._open_image(idx)

        item = {
            "id": self.get_id(idx),
            "image": image,
            "keypoints": [],  # expected by albumentations
        }
        if ce is not None:
            item["ce"] = ce

        if self.transform is not None:
            item = self.transform(**item)

        if normalize:
            item = self.normalize(**item)

        if "ce" in item:
            item["image"] = np.concatenate([item["image"], item.pop("ce")], axis=-1)

        item = to_tensor(**item)

        return item

    def __getitem__(self, idx):
        try:
            return self.getitem(idx, normalize=True)
        except Exception as ex:
            if self.ignore_exceptions:
                warnings.warn(f"Exception with image {self.get_id(idx)}: {ex}")
                return None
            else:
                # raise RuntimeError(f"Exception with image {self.get_id(idx)}") from ex
                print(f"Exception with image {self.get_id(idx)}: {ex}")
                return None
