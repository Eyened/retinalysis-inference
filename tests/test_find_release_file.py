from pathlib import Path

import pytest

from rtnls_inference.utils import find_release_file


def test_find_release_file_legacy_flat(tmp_path: Path):
    release_path = tmp_path / "vessels_may26"
    flat = tmp_path / "vessels_may26.pt"
    flat.write_bytes(b"pt")
    assert find_release_file(release_path) == flat


def test_find_release_file_nested_folder(tmp_path: Path):
    release_path = tmp_path / "vessels_may26"
    release_path.mkdir()
    nested = release_path / "vessels_may26.pt"
    nested.write_bytes(b"pt")
    assert find_release_file(release_path) == nested


def test_find_release_file_prefers_legacy_when_both_exist(tmp_path: Path):
    release_path = tmp_path / "vessels_may26"
    release_path.mkdir()
    flat = tmp_path / "vessels_may26.pt"
    nested = release_path / "vessels_may26.pt"
    flat.write_bytes(b"flat")
    nested.write_bytes(b"nested")
    assert find_release_file(release_path) == flat


def test_find_release_file_nested_onnx(tmp_path: Path):
    release_path = tmp_path / "vessels_may26"
    release_path.mkdir()
    nested = release_path / "vessels_may26.onnx"
    nested.write_bytes(b"onnx")
    assert find_release_file(release_path) == nested


def test_find_release_file_missing_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="No release file found"):
        find_release_file(tmp_path / "missing_model")
