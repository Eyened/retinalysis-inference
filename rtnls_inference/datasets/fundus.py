import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from rtnls_inference.artery_vein import load_av_head_logits
from rtnls_inference.readers import BinaryMaskReader, MaskReader
from rtnls_inference.transforms.base import TestTransform
from rtnls_inference.utils import (
    format_keypoints_for_transform,
    format_keypoints_to_tensor,
    load_image,
)

from .base import TestDataset


class FundusTestDataset(TestDataset):
    def __init__(
        self,
        data: Dict[str, Any],
        transform: Optional[TestTransform] = None,
        ignore_exceptions: bool = False,
        mask_reader: Optional[MaskReader] = None,
        input_mask_reader: Optional[MaskReader] = None,
        loss_mask_reader: Optional[MaskReader] = None,
        num_classes: Optional[int] = None,
        keypoint_names: Optional[List[str]] = None,
        **kwargs,
    ):
        self.data = data
        self.transform = transform
        self.ignore_exceptions = ignore_exceptions
        self.mask_reader = mask_reader or BinaryMaskReader()
        self.input_mask_reader = input_mask_reader or BinaryMaskReader()
        self.loss_mask_reader = loss_mask_reader or BinaryMaskReader()
        self.num_classes = num_classes
        self.keypoint_names = keypoint_names

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

    def _open_logits(self, idx):
        """Open optional teacher logits."""
        item = self.data["images"][idx]
        fpath = item.get("logits")
        if fpath is None:
            return None

        return np.load(fpath).astype(np.float32, copy=False)

    def _open_head_logits(self, idx):
        entry = self.data["images"][idx]
        fpath = entry.get("head_logits")
        if fpath is None:
            return None
        if entry.get("mask") is not None:
            raise ValueError("mask and head_logits are mutually exclusive")
        return load_av_head_logits(fpath)

    def _open_masks_multilabel(self, idx):
        """Open named binary masks as a channel-last mask stack."""
        entry = self.data["images"][idx]
        masks_multilabel = entry.get("masks_multilabel")
        if masks_multilabel is None:
            return None

        if not isinstance(masks_multilabel, dict):
            raise TypeError("masks_multilabel must be a dict of name -> path")

        masks = [self.loss_mask_reader(fpath) for fpath in masks_multilabel.values()]
        return np.stack(masks, axis=-1)

    def _format_masks_multilabel(self, masks):
        """Convert transformed multilabel masks to channel-first float tensors."""
        if not torch.is_tensor(masks):
            masks = torch.as_tensor(masks)
        if masks.ndim != 3:
            raise ValueError(f"masks_multilabel must have 3 dimensions, got {masks.shape}")
        if masks.shape[0] > masks.shape[-1]:
            masks = masks.permute(2, 0, 1)
        return masks.float()

    def _format_logits(self, logits):
        """Convert transformed teacher logits to channel-first float tensors."""
        if not torch.is_tensor(logits):
            logits = torch.as_tensor(logits)
        if logits.ndim != 3:
            raise ValueError(f"logits must have 3 dimensions, got {logits.shape}")

        if self.num_classes is not None:
            if logits.shape[-1] == self.num_classes:
                logits = logits.permute(2, 0, 1)
            elif logits.shape[0] != self.num_classes:
                raise ValueError(
                    f"logits must have {self.num_classes} channels, got {logits.shape}"
                )
        elif logits.shape[-1] <= 16 and logits.shape[0] > 16:
            logits = logits.permute(2, 0, 1)

        return logits.float()

    @staticmethod
    def _format_head_logits(logits):
        if not torch.is_tensor(logits):
            logits = torch.as_tensor(logits)
        if logits.ndim != 3:
            raise ValueError(f"head_logits must have 3 dimensions, got {logits.shape}")
        if logits.shape[-1] == 7:
            logits = logits.permute(2, 0, 1)
        elif logits.shape[0] != 7:
            raise ValueError(f"head_logits must have seven channels, got {logits.shape}")
        return logits.float()

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

    def _open_loss_mask(self, idx):
        """Open optional loss mask, expected to be binary."""
        entry = self.data["images"][idx]
        fpath = entry.get("loss_mask")
        if fpath is None:
            return None

        mask = self.loss_mask_reader(fpath)
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
        logits = self._open_logits(idx)
        head_logits = self._open_head_logits(idx)
        masks_multilabel = self._open_masks_multilabel(idx)
        input_mask = self._open_input_mask(idx)
        loss_mask = self._open_loss_mask(idx)
        entry = self.data["images"][idx]

        item = {
            "id": self.get_id(idx),
            "image": image,
            "mask": mask,
            "logits": logits,
            "head_logits": head_logits,
            "masks_multilabel": masks_multilabel,
            "input_mask": input_mask,
            "loss_mask": loss_mask,
            "labels": entry.get("labels", None),
            "keypoints": format_keypoints_for_transform(
                entry.get("keypoints", []),
                self.keypoint_names,
            ),
            "crops": entry.get("crops", None),
            "metadata": entry.get("metadata", {}),
        }

        if ce is not None:
            item["ce"] = ce

        # Filter None values (except keypoints which we ensured is list)
        item = {k: v for k, v in item.items() if v is not None}

        if self.transform is not None:
            item = self.transform(**item)

        if "keypoints" in item:
            item["keypoints"] = format_keypoints_to_tensor(item["keypoints"])

        if "masks_multilabel" in item:
            item["masks_multilabel"] = self._format_masks_multilabel(
                item["masks_multilabel"]
            )

        if "logits" in item:
            item["logits"] = self._format_logits(item["logits"])

        if "head_logits" in item:
            item["head_logits"] = self._format_head_logits(item["head_logits"])

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
