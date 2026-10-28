from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def project_root(tmp_path):
    """Create the package layout used by project-local components."""
    for relative in (
        ".",
        "rtnls_components",
        "rtnls_components/transforms",
        "rtnls_components/transforms/test",
        "rtnls_components/datasets",
    ):
        directory = tmp_path / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").touch()
    return tmp_path


def _run(cwd: Path, code: str, *, pythonpath: tuple[Path, ...] = ()):
    environment = os.environ.copy()
    package_root = str(Path(__file__).resolve().parents[1])
    # Make fixture packages importable without installing them into the environment.
    environment["PYTHONPATH"] = os.pathsep.join(
        [package_root, *map(str, pythonpath), environment.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_transform(path: Path, class_name: str, extra: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "from rtnls_inference.transforms.base import TestTransform\n"
        f"{extra}\n"
        f"class {class_name}(TestTransform):\n"
        "    def _transform(self, preprocess=None, **item): return item\n"
        "    def undo_item(self, item, preprocess=False): return item\n"
    )


def test_packaged_components_import_once_in_sorted_order(project_root):
    log = project_root / "imports.txt"
    component_root = project_root / "rtnls_components" / "transforms" / "test"
    _write_transform(
        component_root / "z_last.py",
        "ZLastTransform",
        f"open({str(log)!r}, 'a').write('z\\n')",
    )
    _write_transform(
        component_root / "a_first.py",
        "AFirstTransform",
        f"open({str(log)!r}, 'a').write('a\\n')",
    )

    result = _run(
        project_root,
        """
        from rtnls_inference.transforms import get_test_transform_registry
        first = get_test_transform_registry()
        second = get_test_transform_registry()
        assert first is second
        assert {'AFirstTransform', 'ZLastTransform'} <= set(first)
        """,
        pythonpath=(project_root.parent,),
    )
    assert result.returncode == 0, result.stderr
    assert log.read_text().splitlines() == ["a", "z"]


def test_discovery_uses_exact_cwd_and_never_searches_parents(project_root):
    _write_transform(
        project_root / "rtnls_components" / "transforms" / "parent.py",
        "ParentOnlyTransform",
    )
    child = project_root / "child"
    child.mkdir()
    result = _run(
        child,
        """
        from rtnls_inference.transforms import get_test_transform_registry
        assert 'ParentOnlyTransform' not in get_test_transform_registry()
        """,
        pythonpath=(project_root.parent,),
    )
    assert result.returncode == 0, result.stderr


def test_import_failure_reports_file_and_preserves_cause(project_root):
    broken = project_root / "rtnls_components" / "datasets" / "broken.py"
    broken.write_text("raise ValueError('project boom')\n")
    result = _run(
        project_root,
        """
        from rtnls_inference.components import import_project_components
        import_project_components()
        """,
        pythonpath=(project_root.parent,),
    )
    assert result.returncode != 0
    assert str(broken) in result.stderr
    assert "ValueError: project boom" in result.stderr


def test_duplicate_component_names_are_fatal(project_root):
    _write_transform(
        project_root / "rtnls_components" / "transforms" / "duplicate.py",
        "FundusTestTransform",
    )
    result = _run(
        project_root,
        """
        from rtnls_inference.transforms import get_test_transform_registry
        get_test_transform_registry()
        """,
        pythonpath=(project_root.parent,),
    )
    assert result.returncode != 0
    assert (
        "Duplicate TestTransform component name 'FundusTestTransform'" in result.stderr
    )


def test_unimportable_project_reports_install_command(project_root):
    result = _run(
        project_root,
        """
        from rtnls_inference.components import import_project_components
        import_project_components()
        """,
    )
    assert result.returncode != 0
    package_name = f"{project_root.name}.rtnls_components"
    assert (
        f"Project components package {package_name!r} is not installed" in result.stderr
    )
    assert f"Run: pip install -e {project_root}" in result.stderr
    assert (
        f"ModuleNotFoundError: No module named {project_root.name!r}" in result.stderr
    )


def test_process_is_bound_to_first_working_directory(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    result = _run(
        tmp_path,
        f"""
        import os
        from rtnls_inference.transforms import get_test_transform_registry
        get_test_transform_registry()
        os.chdir({str(other)!r})
        get_test_transform_registry()
        """,
    )
    assert result.returncode != 0
    assert "already initialized" in result.stderr
    assert "Start a new Python process" in result.stderr
