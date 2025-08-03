import albumentations as A
import cv2
import torch

from rtnls_fundusprep.mask_extraction import CFIBounds as Bounds
from rtnls_fundusprep.mask_extraction import get_cfi_bounds
from rtnls_fundusprep.preprocessor import FundusPreprocessor

from .base import TestTransform


class FundusTestTransform(TestTransform):
    def __init__(
        self,
        square_size=1024,
        resize=None,
        preprocess=False,
        contrast_enhance=True,
        **kwargs,
    ):
        self.prep_function = FundusPreprocessor(
            square_size=square_size,
        )
        if resize is None:
            resize = square_size
        self.resize = resize
        self.square_size = square_size
        self.preprocess = preprocess
        self.contrast_enhance = contrast_enhance
        self.transform = A.Compose(
            [A.Resize(resize, resize)],
            additional_targets={"ce": "image"},
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

        print('FundusTestTransform initialized with contrast_enhance:', self.contrast_enhance)

    def undo_resize(self, proba):
        return cv2.resize(
            proba,
            (self.square_size, self.square_size),
            interpolation=cv2.INTER_LINEAR,
        )

    def undo_item(self, item, preprocess=None):
        do_preprocess = preprocess if preprocess is not None else self.preprocess

        new_item = {**item}
        if "image" in item:
            image = self.undo_resize(item["image"])
            if do_preprocess:
                bounds = Bounds(**item["metadata"]["bounds"])
                M = bounds.get_cropping_matrix(self.square_size)
                new_item["image"] = M.warp_inverse(image, (bounds.h, bounds.w))
            else:
                new_item["image"] = image

        if "keypoints" in item:
            kp = (self.square_size / self.resize) * item["keypoints"]
            if do_preprocess:
                bounds = Bounds(**item["metadata"]["bounds"])
                M = bounds.get_cropping_matrix(self.square_size)
                new_item["keypoints"] = M.apply_inverse(kp)
            else:
                new_item["keypoints"] = kp
        return new_item

    def __call__(self, preprocess=None, **item):
        if "ce" in item and self.contrast_enhance:
            raise ValueError(
                "Contrast enhancement image already present in kwargs. Would apply contrast enhancement twice."
            )

        do_preprocess = preprocess if preprocess is not None else self.preprocess
        if do_preprocess:
            # we preprocess without contrast enhance
            # to add more logic to this part
            item = self.prep_function(**item)

        if self.contrast_enhance:
            # if the bounds of the original (non-cropped) image are available
            if "bounds" in item["metadata"]:
                _, bounds = Bounds(**item["metadata"]["bounds"]).crop(self.square_size)
            else:  # else we compute the bounds of the provided image
                bounds = get_cfi_bounds(item["image"])

            item["ce"] = bounds.contrast_enhanced_5

        # serialize the bounds
        # cannot pass arbitrary objects to the dataloader
        # if "bounds" in item:
        #     item["metadata"]["bounds"] = item["bounds"].to_dict()

        item = self.transform(**item)
        return item


def extract_bound(d, i):
    if isinstance(d, dict):
        return {k: extract_bound(v, i) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        return d[i].item()
    else:
        return d
