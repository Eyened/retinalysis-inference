from rtnls_inference.utils import get_all_subclasses_dict

from .base import TestTransform
from .fundus import FundusTestTransform

test_transforms = get_all_subclasses_dict(TestTransform)


def make_test_transform(transform_cfg, **kwargs):
    class_name = transform_cfg.get("class", None)
    if class_name is None:
        return FundusTestTransform(**transform_cfg, **kwargs)
    
    test_transform_class = test_transforms.get(
        class_name, None
    )

    if test_transform_class is None:
        return None

    return test_transform_class(**transform_cfg, **kwargs)
