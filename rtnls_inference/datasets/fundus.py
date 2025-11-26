import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

from rtnls_inference.transforms.base import TestTransform
from rtnls_inference.utils import load_image

from .base import TestDataset


class FundusTestDataset(TestDataset):
    def __init__(
        self,
        data: Dict,
        transform: TestTransform = None,
        ignore_exceptions: bool = False,
        mask_reader: Optional[Callable] = None,
        overlapping_masks: bool = False,
        num_classes: Optional[int] = None,
        mask_label: Optional[int] = None,
        **kwargs,
    ):
        self.data = data
        self.transform = transform
        self.ignore_exceptions = ignore_exceptions
        self.mask_reader = mask_reader or self._default_mask_reader
        self.overlapping_masks = overlapping_masks
        self.num_classes = num_classes
        self.mask_label = mask_label

    def _default_mask_reader(self, fpath: Union[str, Path]):
        if isinstance(fpath, str):
            fpath = Path(fpath)
        if fpath.suffix == ".npy":
            return (np.load(fpath)).astype(np.uint8)
        else:
            image = Image.open(fpath)
        im = np.array(image, dtype=np.uint8)
        return im

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
        """Open label (segmentation) masks optionally containing a loss mask (to be zeroed out)
        It always returns one-hot encoded label masks with the loss mask as the last channel
        """
        item = self.data["images"][idx]
        fpath = item.get("mask")
        if fpath is None:
            return None

        mask = self.mask_reader(fpath)
        if not self.overlapping_masks:
            # mask contains class labels
            if len(mask.squeeze().shape) != 2:
                # It might be that the mask reader returned something with channels
                # If it's single channel, squeeze it.
                if mask.ndim == 3 and mask.shape[2] == 1:
                    mask = mask.squeeze()
                elif mask.ndim == 3 and mask.shape[0] == 1:  # CHW? unlikely for PIL
                    mask = mask.squeeze()

            # Re-check shape
            if len(mask.squeeze().shape) != 2:
                # Just return as is if we can't figure it out, but warn?
                # For now assume it's correct or let downstream fail
                pass

            if self.num_classes is not None:
                assert np.max(mask) >= self.num_classes, (
                    f"Too many classes found in mask file {fpath}"
                )

            if self.mask_label is not None:
                new_mask = mask.copy()
                new_mask[new_mask == self.mask_label] = 0
                masks = [new_mask, (mask == self.mask_label).astype(np.uint8)]
            else:
                masks = [mask, np.zeros_like(mask)]

        else:
            # overlapping masks
            assert len(mask.squeeze().shape) == 3, (
                f"Invalid mask shape {mask.squeeze().shape}"
            )

            masks = np.split(mask, mask.shape[-1], axis=-1)
            masks = [mask.squeeze() for mask in masks]

        return masks

    def get_id(self, idx):
        return self.data["images"][idx]["id"]

    def getitem(self, idx):
        image, ce = self._open_image(idx)
        masks = self._open_mask(idx)
        entry = self.data["images"][idx]

        item = {
            "id": self.get_id(idx),
            "image": image,
            "masks": masks,
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

        # After transform, images are Tensors (CHW).
        # We need to handle 'ce' if it's still separate (BasicTestTransform puts 'ce' in additional_targets)
        # The transform handles 'ce' so it should be transformed and tensorized if it was passed.

        if "ce" in item and isinstance(item["ce"], (np.ndarray, list)):
            # If transform didn't tensorize it (e.g. no transform), we might need to?
            # But we assume transform does it.
            pass

        if "ce" in item:
            # Concatenate CE if present (similar to FundusDataset logic)
            # FundusDataset: item["image"] = np.concatenate([item["image"], item.pop("ce")], axis=axis)
            # Axis depends on normalization strategy.
            # If tensor (CHW), axis=0. If numpy (HWC), axis=-1.
            # Here we expect Tensors from transform.
            if hasattr(item["image"], "shape") and hasattr(item["ce"], "shape"):
                # Check if they are tensors
                if hasattr(item["image"], "device"):  # simple check for tensor
                    item["image"] = torch.cat([item["image"], item.pop("ce")], dim=0)
                else:
                    # numpy
                    item["image"] = np.concatenate(
                        [item["image"], item.pop("ce")], axis=-1
                    )

        if "masks" in item and len(item["masks"]) > 0:
            # FundusDataset logic for masks
            item["loss_mask"] = item["masks"].pop()
            # masks is a list of masks.
            # If transform processed masks, they might be tensors or list of tensors?
            # 'masks' in item passed to transform: list of numpy arrays.
            # Albumentations 'masks' target expects list of images.
            # So they come back as list of transformed masks.

            masks = item["masks"]
            if isinstance(masks, list) and len(masks) > 0:
                # Stack them.
                # If tensors (from ToTensorV2 which doesn't affect masks usually unless configured?)
                # ToTensorV2 usually affects 'image' and 'mask' (if single) or 'masks' (if list).
                # Wait, ToTensorV2 in albumentations:
                # "Convert image and mask to `torch.Tensor`."
                # It converts 'mask' or 'masks'.

                if isinstance(masks[0], (np.ndarray, np.generic)):
                    item["masks"] = np.stack(masks, axis=-1).transpose(2, 0, 1)  # CHW
                    # Convert to tensor if image is tensor?
                    if hasattr(item["image"], "device"):
                        item["masks"] = torch.from_numpy(item["masks"])
                elif hasattr(masks[0], "device"):  # tensor
                    item["masks"] = torch.stack(masks, dim=0)

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
