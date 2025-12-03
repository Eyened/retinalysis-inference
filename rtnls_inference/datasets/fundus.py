import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch

from rtnls_inference.readers import BinaryMaskReader, MaskReader
from rtnls_inference.transforms.base import TestTransform
from rtnls_inference.utils import load_image

from .base import TestDataset


class FundusTestDataset(TestDataset):
    def __init__(
        self,
        data: Dict[str, Any],
        transform: Optional[TestTransform] = None,
        ignore_exceptions: bool = False,
        mask_reader: Optional[MaskReader] = None,
        input_mask_reader: Optional[MaskReader] = None,
        num_classes: Optional[int] = None,
        **kwargs,
    ):
        self.data = data
        self.transform = transform
        self.ignore_exceptions = ignore_exceptions
        self.mask_reader = mask_reader or BinaryMaskReader()
        self.input_mask_reader = input_mask_reader or BinaryMaskReader()
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data["images"])

    def _open_image(self, idx):
        entry = self.data["images"][idx]
        image_path = entry.get("image")
        ce_path = entry.get("contrast_enhanced") or entry.get("ce")  # support both keys

        image = load_image(image_path) if image_path else None
        ce = load_image(ce_path) if ce_path else None
        return image, ce

    def _open_mask(self, idx):
        """Open label (segmentation) masks"""
        item = self.data["images"][idx]
        fpath = item.get("mask")
        if fpath is None:
            return None

        mask = self.mask_reader(fpath)

        return mask

    def _open_input_mask(self, idx):
        """Open optional input mask, expected to be binary."""
        entry = self.data["images"][idx]
        fpath = entry.get("input_mask")
        if fpath is None:
            return None

        mask = self.input_mask_reader(fpath)
        if isinstance(mask, np.ndarray) and mask.ndim == 3:
            # Squeeze singleton channel dimensions if present
            if mask.shape[0] == 1 or mask.shape[-1] == 1:
                mask = np.squeeze(mask)
        return mask

    def get_id(self, idx):
        return self.data["images"][idx]["id"]

    def getitem(self, idx):
        image, ce = self._open_image(idx)
        mask = self._open_mask(idx)
        input_mask = self._open_input_mask(idx)
        entry = self.data["images"][idx]

        item = {
            "id": self.get_id(idx),
            "image": image,
            "mask": mask,
            "input_mask": input_mask,
            "labels": entry.get("labels", None),
            "keypoints": entry.get("keypoints", []),  # expected by albumentations
            "crops": entry.get("crops", None),
            "metadata": entry.get("metadata", {}),
        }

        if ce is not None:
            item["ce"] = ce

        # Filter None values (except keypoints which we ensured is list)
        item = {k: v for k, v in item.items() if v is not None}

        if self.transform is not None:
            item = self.transform(**item)

        if "ce" in item:
            # Assuming tensor output from transform (CHW)
            item["image"] = torch.cat([item["image"], item.pop("ce")], dim=0)

        return item

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)

        except Exception as ex:
            if self.ignore_exceptions:
                warnings.warn(f"Exception with image {self.get_id(idx)}: {ex}")
                return None
            else:
                raise RuntimeError(f"Exception with image {self.get_id(idx)}") from ex
