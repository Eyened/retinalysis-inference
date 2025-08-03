from rtnls_inference.utils import get_all_subclasses_dict

from .base import TestTransform
from .fundus import FundusTestTransform

test_transforms = get_all_subclasses_dict(TestTransform)


def make_test_transform(config, **kwargs):
    test_cfg = config["datamodule"].get("test_transform", {})
    if "test_transform" not in config["datamodule"]:
        return FundusTestTransform(**{**test_cfg, **kwargs})

    class_name = test_cfg.get("class", None)
    if class_name is None:
        return FundusTestTransform(**{**test_cfg, **kwargs})
    
    test_transform_class = test_transforms.get(
        class_name, None
    )

    if test_transform_class is None:
        return None

    args = {**test_cfg, **kwargs}
    args["base_path"] = config["datamodule"].get("base_path", None)
    return test_transform_class(**args)
