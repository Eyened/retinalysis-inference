import warnings
from typing import Any, Dict, Optional

import torch

from rtnls_inference.dtos_inference import InferenceDatasetDTO, ModelInputDTO
from rtnls_inference.readers import BinaryMaskReader, MaskReader
from rtnls_inference.transforms.base import TestTransform
from rtnls_inference.utils import load_image

from .base import TestDataset


class FundusInferenceDataset(TestDataset):
    """Strict fundus prediction dataset accepting only ``ModelInputDTO`` fields."""

    item_dto_cls = ModelInputDTO
    dataset_dto_cls = InferenceDatasetDTO[ModelInputDTO]

    def __init__(
        self,
        data: Dict[str, Any],
        transform: Optional[TestTransform] = None,
        ignore_exceptions: bool = False,
        input_mask_reader: Optional[MaskReader] = None,
        **kwargs,
    ):
        definition = self.dataset_dto_cls.model_validate(data)
        self.data = definition.to_serialized_images()
        self.transform = transform
        self.ignore_exceptions = ignore_exceptions
        self.input_mask_reader = input_mask_reader or BinaryMaskReader()

    def __len__(self):
        return len(self.data)

    def get_id(self, idx):
        return self.data[idx]["id"]

    def _open_image(self, entry):
        image_path = entry.get("image")
        ce_path = entry.get("contrast_enhanced")
        image = load_image(image_path) if image_path else None
        ce = load_image(ce_path) if ce_path else None
        return image, ce

    def _open_input_mask(self, entry):
        path = entry.get("input_mask")
        return self.input_mask_reader(path) if path else None

    def getitem(self, idx):
        entry = self.data[idx]
        image, ce = self._open_image(entry)
        item = {
            "id": entry["id"],
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
        return item

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception as exc:
            if self.ignore_exceptions:
                warnings.warn(f"Exception with image {self.get_id(idx)}: {exc}")
                return None
            raise RuntimeError(f"Exception with image {self.get_id(idx)}") from exc
