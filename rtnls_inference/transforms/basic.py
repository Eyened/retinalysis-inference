import albumentations as A

from .base import TestTransform


class BasicTestTransform(TestTransform):
    def __init__(self, size=256, pad: [int, int] = None, normalize="imagenet") -> None:
        super().__init__(normalize=normalize)

        transforms = [
            A.PadIfNeeded(
                min_height=pad[0], min_width=pad[1], border_mode=0, value=(0, 0, 0)
            )
            if pad is not None
            else A.NoOp(),
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(
                min_height=size, min_width=size, border_mode=0, value=(0, 0, 0)
            ),
        ]

        self.transform = A.Compose(
            transforms,
            additional_targets={
                "ce": "image",
                "input_mask": "mask",
                "loss_mask": "mask",
                "masks_multilabel": "mask",
            },
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

    def undo_item(self, item, preprocess=False):
        return item

    def _transform(self, preprocess=None, **item):
        return self.transform(**item)
