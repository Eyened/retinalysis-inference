from abc import abstractmethod
from typing import Any, Dict

import albumentations as A
from albumentations.pytorch import ToTensorV2


class TestTransform:
    def __init__(self, normalize="imagenet"):
        transforms = []
        if normalize == "imagenet":
            transforms.append(
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                )
            )
        elif normalize == "diffusion":
            transforms.append(
                A.Normalize(
                    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0
                )
            )
        elif normalize == "zero-one":
            transforms.append(
                A.Normalize(
                    mean=(0.0, 0.0, 0.0),
                    std=(1.0, 1.0, 1.0),
                    max_pixel_value=255.0,
                )
            )
        elif normalize:
            raise ValueError(f"Invalid normalization strategy: {normalize}")

        transforms.append(ToTensorV2())

        self.post_transform = A.Compose(
            transforms,
            additional_targets={
                "ce": "image",
                "input_mask": "mask",
                "loss_mask": "mask",
                "masks_multilabel": "mask",
            },
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

    def add_targets(self, mapping):
        """Add dataset-owned targets to every Albumentations composition."""
        if not mapping:
            return
        visited = set()

        def bind(value):
            if id(value) in visited:
                return
            visited.add(id(value))
            if isinstance(value, A.BaseCompose):
                value.add_targets(mapping)
            elif isinstance(value, dict):
                for child in value.values():
                    bind(child)
            elif isinstance(value, (list, tuple, set)):
                for child in value:
                    bind(child)

        for value in vars(self).values():
            bind(value)

    @abstractmethod
    def undo_item(self, item: Dict[str, Any], preprocess: bool = False):
        pass

    @abstractmethod
    def _transform(self, preprocess: bool = None, **item):
        pass

    def __call__(self, preprocess: bool = None, **item):
        item = self._transform(preprocess=preprocess, **item)
        preprocessed_image = item.pop("preprocessed_image", None)
        item.setdefault("keypoints", [])
        item = self.post_transform(**item)
        if preprocessed_image is not None:
            item["preprocessed_image"] = preprocessed_image
        return item
