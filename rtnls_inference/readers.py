import warnings
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from rtnls_inference.utils import get_all_subclasses_dict


class MaskReader:
    """
    Base class for mask readers. Subclasses should return a class map of shape (H, W).
    """

    def __init__(self, one_hot=False, num_classes=None, **kwargs):
        self.one_hot = one_hot
        self.num_classes = num_classes

    def __call__(self, fpath: Union[str, Path]):
        mask = self.read_mask(fpath)
        if mask.ndim != 2:
            raise ValueError(
                f"MaskReader expected output shape (H, W), got {mask.shape} for file {fpath}"
            )
        return mask

    def read_mask(self, fpath: Union[str, Path]):
        raise NotImplementedError


class DefaultMaskReader(MaskReader):
    def read_mask(self, fpath: Union[str, Path]):
        if isinstance(fpath, str):
            fpath = Path(fpath)
        image = Image.open(fpath)
        im = np.array(image, dtype=np.uint8)
        if im.ndim == 3:
            im = im[..., 0]
        return im


class BinaryMaskReader(MaskReader):
    def read_mask(self, fpath: Union[str, Path]):
        if isinstance(fpath, str):
            fpath = Path(fpath)
        image = Image.open(fpath)
        im = np.array(image, dtype=np.uint8)
        if im.ndim == 3:
            im = im[..., 0]
        if len(np.unique(im)) > 2:
            warnings.warn(f"Found {len(np.unique(im))} unique values in image {fpath}")
        # if np.array_equal(np.unique(im), [0, 255]):
        #     im[im == 255] = 1
        im[im > 0] = 1
        return im


class ArteriesReader(MaskReader):
    def read_mask(self, fpath: Union[str, Path]):
        if isinstance(fpath, str):
            fpath = Path(fpath)
        image = Image.open(fpath)
        im = np.array(image, dtype=np.uint8)
        if im.ndim == 3:
            im = im[..., 0]

        return ((im == 1) | (im == 3)).astype(np.uint8)


class VeinsReader(MaskReader):
    def read_mask(self, fpath: Union[str, Path]):
        if isinstance(fpath, str):
            fpath = Path(fpath)
        image = Image.open(fpath)
        im = np.array(image, dtype=np.uint8)
        if im.ndim == 3:
            im = im[..., 0]
        return ((im == 2) | (im == 3)).astype(np.uint8)


readers = get_all_subclasses_dict(MaskReader)


def make_mask_reader(config):
    if config is None:
        config = {}
    reader_class = readers.get(config.get("class", None), DefaultMaskReader)

    return reader_class(**config)
