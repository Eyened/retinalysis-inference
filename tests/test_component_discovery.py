from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def _run(cwd: Path, code: str):
    environment = os.environ.copy()
    package_root = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = os.pathsep.join(
        [package_root, environment.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
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


def test_package_free_components_import_once_in_sorted_order(tmp_path):
    log = tmp_path / "imports.txt"
    component_root = tmp_path / "rtnls_components" / "transforms" / "test"
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
        tmp_path,
        """
        from rtnls_inference.transforms import get_test_transform_registry
        first = get_test_transform_registry()
        second = get_test_transform_registry()
        assert first is second
        assert {'AFirstTransform', 'ZLastTransform'} <= set(first)
        """,
    )
    assert result.returncode == 0, result.stderr
    assert log.read_text().splitlines() == ["a", "z"]
    assert not list((tmp_path / "rtnls_components").rglob("__init__.py"))


def test_discovery_uses_exact_cwd_and_never_searches_parents(tmp_path):
    _write_transform(
        tmp_path / "rtnls_components" / "transforms" / "parent.py",
        "ParentOnlyTransform",
    )
    child = tmp_path / "child"
    child.mkdir()
    result = _run(
        child,
        """
        from rtnls_inference.transforms import get_test_transform_registry
        assert 'ParentOnlyTransform' not in get_test_transform_registry()
        """,
    )
    assert result.returncode == 0, result.stderr


def test_import_failure_reports_file_and_preserves_cause(tmp_path):
    broken = tmp_path / "rtnls_components" / "datasets" / "broken.py"
    broken.parent.mkdir(parents=True)
    broken.write_text("raise ValueError('project boom')\n")
    result = _run(
        tmp_path,
        """
        from rtnls_inference.components import import_project_components
        import_project_components()
        """,
    )
    assert result.returncode != 0
    assert str(broken) in result.stderr
    assert "ValueError: project boom" in result.stderr


def test_duplicate_component_names_are_fatal(tmp_path):
    _write_transform(
        tmp_path / "rtnls_components" / "transforms" / "duplicate.py",
        "FundusTestTransform",
    )
    result = _run(
        tmp_path,
        """
        from rtnls_inference.transforms import get_test_transform_registry
        get_test_transform_registry()
        """,
    )
    assert result.returncode != 0
    assert "Duplicate TestTransform component name 'FundusTestTransform'" in result.stderr


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
