"""Project-local component discovery shared by training and inference.

Projects must be installed (``pip install -e .`` from the project root).
Discovery imports ``{cwd.name}.rtnls_components`` and walks its submodules so
class bodies run before registries snapshot ``__subclasses__()``.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import TypeVar


_initialized_cwd: Path | None = None
_project_components_imported = False

T = TypeVar("T", bound=type)


def _bind_working_directory(cwd: Path | None = None) -> Path:
    global _initialized_cwd

    resolved = (cwd or Path.cwd()).resolve()
    if _initialized_cwd is None:
        _initialized_cwd = resolved
    elif _initialized_cwd != resolved:
        raise RuntimeError(
            "Retinalysis component discovery is already initialized for "
            f"{_initialized_cwd}; refusing to reuse it from {resolved}. "
            "Start a new Python process when changing project roots."
        )
    return resolved


def import_project_components(cwd: Path | None = None) -> Path | None:
    """Import ``{cwd.name}.rtnls_components`` and its submodules exactly once."""
    global _project_components_imported

    project_root = _bind_working_directory(cwd)
    component_root = project_root / "rtnls_components"
    if _project_components_imported:
        return component_root if component_root.is_dir() else None

    if not component_root.is_dir():
        _project_components_imported = True
        return None

    package_name = f"{project_root.name}.rtnls_components"
    try:
        package = importlib.import_module(package_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Project components package {package_name!r} is not installed. "
            f"Run: pip install -e {project_root}"
        ) from exc

    for module_info in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        try:
            importlib.import_module(module_info.name)
        except Exception as walk_exc:
            raise RuntimeError(
                f"Failed to import project component {module_info.name}: {walk_exc}"
            ) from walk_exc

    _project_components_imported = True
    return component_root


def build_subclass_registry(base_class: type) -> dict[str, type]:
    """Build a duplicate-safe registry of concrete recursive subclasses."""
    registry: dict[str, type] = {}
    seen: set[type] = set()

    def visit(parent: type) -> None:
        for subclass in parent.__subclasses__():
            if subclass in seen:
                continue
            seen.add(subclass)
            visit(subclass)
            if inspect.isabstract(subclass):
                continue
            previous = registry.get(subclass.__name__)
            if previous is not None and previous is not subclass:
                raise RuntimeError(
                    f"Duplicate {base_class.__name__} component name "
                    f"{subclass.__name__!r}: {previous.__module__}.{previous.__qualname__} "
                    f"and {subclass.__module__}.{subclass.__qualname__}"
                )
            registry[subclass.__name__] = subclass

    visit(base_class)
    return registry


def initialized_working_directory() -> Path | None:
    """Return the process-bound discovery directory, primarily for diagnostics."""
    return _initialized_cwd
