import albumentations as A
import cv2
import torch
from rtnls_fundusprep.mask_extraction import Bounds, extract_bounds
from rtnls_fundusprep.preprocessor import FundusPreprocessor, contrast_enhance

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
        )

    def undo_resize(self, proba):
        return cv2.resize(
            proba,
            (self.square_size, self.square_size),
            interpolation=cv2.INTER_LINEAR,
        )

    # def undo(self, batch, proba, preprocess=False):
    #     proba = self.undo_resize(proba)
    #     do_preprocess = preprocess if preprocess is not None else self.preprocess
    #     if not do_preprocess:
    #         return proba

    #     unprep = []

    #     for i in range(batch["image"].shape[0]):
    #         bounds = Bounds.from_dict(extract_bound(batch["bounds"], i))
    #         M = bounds.get_cropping_matrix(self.square_size)
    #         im = proba[i, ...].squeeze()
    #         unprep.append(M.warp_inverse(im, (bounds.h, bounds.w)))

    #     return unprep

    # def undo_keypoints(self, batch, kp, preprocess=False):
    #     kp = (self.square_size) * kp

    #     do_preprocess = preprocess if preprocess is not None else self.preprocess
    #     if not do_preprocess:
    #         return kp

    #     unprep = []

    #     for i in range(batch["image"].shape[0]):
    #         bounds = Bounds.from_dict(extract_bound(batch["bounds"], i))
    #         M = bounds.get_cropping_matrix(self.square_size)
    #         im_kp = kp[i, ...]
    #         unprep.append(M.apply_inverse(im_kp, (bounds.h, bounds.w)))

    #     return np.stack(unprep)

    def undo_item(self, item, preprocess=False):
        do_preprocess = preprocess if preprocess is not None else self.preprocess

        new_item = {**item}
        if "image" in item:
            image = self.undo_resize(item["image"])
            if do_preprocess:
                bounds = Bounds.from_dict(item["bounds"])
                M = bounds.get_cropping_matrix(self.square_size)
                new_item["image"] = M.warp_inverse(image, (bounds.h, bounds.w))
            else:
                new_item["image"] = image

        if "keypoints" in item:
            kp = (self.square_size) * item["keypoints"]
            if do_preprocess:
                bounds = Bounds.from_dict(item["bounds"])
                M = bounds.get_cropping_matrix(self.square_size)
                new_item["keypoints"] = M.apply_inverse(kp, (bounds.h, bounds.w))
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
            item = self.prep_function(**item)

        if self.contrast_enhance:
            # if the bounds are available
            if "bounds" in item:
                M = item["bounds"].get_cropping_matrix(self.square_size)
                bounds = item["bounds"].warp(M, (self.square_size, self.square_size))
            else:  # else we compute the bounds
                bounds = extract_bounds(item["image"])
            mask = bounds.make_binary_mask(0.01 * bounds.radius)
            mirrored = bounds.background_mirroring(item["image"])
            sigma = 0.05 * bounds.radius

            item["ce"] = contrast_enhance(mirrored, mask, sigma)

        # serialize the bounds
        # cannot pass arbitrary objects to the dataloader
        if "bounds" in item:
            item["bounds"] = item["bounds"].to_dict()

        item = self.transform(**item)
        return item


def extract_bound(d, i):
    if isinstance(d, dict):
        return {k: extract_bound(v, i) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        return d[i].item()
    else:
        return d
