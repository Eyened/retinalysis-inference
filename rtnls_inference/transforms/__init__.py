from rtnls_inference.components import (
    build_subclass_registry,
    import_project_components,
)

from .base import TestTransform
from .basic import BasicTestTransform  # noqa: F401
from .fundus import FundusTestTransform

_test_transforms: dict[str, type[TestTransform]] | None = None


def get_test_transform_registry() -> dict[str, type[TestTransform]]:
    global _test_transforms
    import_project_components()
    if _test_transforms is None:
        _test_transforms = build_subclass_registry(TestTransform)
    return _test_transforms


def make_test_transform(transform_cfg, **kwargs):
    class_name = transform_cfg.get("class", None)
    if class_name is None:
        return FundusTestTransform(**transform_cfg, **kwargs)
    
    test_transform_class = get_test_transform_registry().get(
        class_name, None
    )

    if test_transform_class is None:
        return None

    return test_transform_class(**transform_cfg, **kwargs)
