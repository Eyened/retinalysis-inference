import albumentations as A

from .base import TestTransform


class BasicTestTransform(TestTransform):
    def __init__(self, size=256) -> None:
        super().__init__()
        self.transform = A.Compose(
            [
                A.LongestMaxSize(max_size=size),
                A.PadIfNeeded(
                    min_height=size, min_width=size, border_mode=0, value=(0, 0, 0)
                ),
            ],
            additional_targets={"ce": "image"},
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

    def undo_item(self, item, preprocess=False):
        return item

    def __call__(self, preprocess=None, **item):
        return self.transform(**item)
