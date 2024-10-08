from abc import abstractmethod
from typing import Any, Dict


class TestTransform:
    @abstractmethod
    def undo_item(self, item: Dict[str, Any], preprocess: bool = False):
        pass

    @abstractmethod
    def __call__(self, preprocess: bool = None, **item):
        pass
