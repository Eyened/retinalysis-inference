import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch

from rtnls_inference.inputs import image_inputs_adapter, loaded_rgb, validate_ids
from rtnls_inference.readers import BinaryMaskReader, MaskReader
from rtnls_inference.transforms.base import TestTransform
from rtnls_inference.utils import load_image

from .base import TestDataset


class FundusInferenceDataset(TestDataset):
    """Strict prediction records with paths or loaded RGB images."""

    def __init__(
        self,
        data: Dict[str, Any],
        transform: Optional[TestTransform] = None,
        ignore_exceptions: bool = False,
        input_mask_reader: Optional[MaskReader] = None,
        **kwargs,
    ):
        if not isinstance(data, dict) or set(data) != {"images"}:
            raise ValueError("Inference data must contain only an 'images' collection")
        self.data = image_inputs_adapter.validate_python(data["images"])
        self.validate_ids()
        self.transform = transform
        self.ignore_exceptions = ignore_exceptions
        self.input_mask_reader = input_mask_reader or BinaryMaskReader()

    def __len__(self):
        return len(self.data)

    def get_id(self, idx):
        return self.data[idx].get("id")

    def validate_ids(self, *, for_output=False):
        validate_ids((entry.get("id") for entry in self.data), for_output=for_output)

    def _open_image(self, entry):
        source = entry.get("image")
        ce_path = entry.get("contrast_enhanced")
        # Give transforms their own pixels so callers' loaded images stay intact.
        image = (
            source.copy()
            if isinstance(source, np.ndarray)
            else (loaded_rgb(load_image(source)) if source else None)
        )
        ce = load_image(ce_path) if ce_path else None
        return image, ce

    def _open_input_mask(self, entry):
        path = entry.get("input_mask")
        return self.input_mask_reader(path) if path else None

    def getitem(self, idx):
        entry = self.data[idx]
        image, ce = self._open_image(entry)
        item = {
            "id": entry.get("id"),
            "image": image,
            "input_mask": self._open_input_mask(entry),
            "metadata": entry.get("metadata"),
            "fov": entry.get("fov"),
        }
        if ce is not None:
            item["ce"] = ce
        item = {key: value for key, value in item.items() if value is not None}
        if self.transform is not None:
            item = self.transform(**item)
        if "input_mask" in item:
            item["input_mask"] = self.input_mask_reader.post_transform(
                item["input_mask"]
            )
        if "ce" in item:
            item["image"] = torch.cat([item["image"], item.pop("ce")], dim=0)
        item["id"] = entry.get("id")
        return item

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception as exc:
            if self.ignore_exceptions:
                warnings.warn(f"Exception with image {self.get_id(idx)}: {exc}")
                return None
            raise RuntimeError(f"Exception with image {self.get_id(idx)}") from exc
