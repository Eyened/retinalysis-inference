import importlib
import json
import shutil
from pathlib import Path

import torch

CONFIG_KEY = "config.yaml"
SUPPORTED_SUFFIXES = {".pt", ".onnx"}


def load_stored_config(path: str | Path) -> dict:
    """Load embedded config from a TorchScript or ONNX release file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".onnx":
        return _load_onnx_config(path)
    if suffix == ".pt":
        return _load_torchscript_config(path)
    raise ValueError(
        f"Unsupported release format {suffix!r} for {path}. "
        f"Expected one of {sorted(SUPPORTED_SUFFIXES)}."
    )


def update_stored_config(
    path: str | Path,
    config: dict,
    *,
    out_path: str | Path | None = None,
) -> Path:
    """Write config into a TorchScript or ONNX release file.

    Only updates stored metadata/extra_files, not model weights or graph structure.
    """
    path = Path(path)
    dest = Path(out_path) if out_path is not None else path

    if dest.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported release format {dest.suffix!r}. "
            f"Expected one of {sorted(SUPPORTED_SUFFIXES)}."
        )

    if dest.resolve() != path.resolve():
        shutil.copy2(path, dest)

    suffix = dest.suffix.lower()
    if suffix == ".onnx":
        _update_onnx_config(dest, config)
    else:
        _update_torchscript_config(dest, config)
    return dest


def _load_onnx_config(path: Path) -> dict:
    onnx = importlib.import_module("onnx")
    model = onnx.load(str(path))
    for prop in model.metadata_props:
        if prop.key == CONFIG_KEY:
            return json.loads(prop.value)
    raise ValueError(f"No {CONFIG_KEY} metadata in {path}")


def _update_onnx_config(path: Path, config: dict) -> None:
    onnx = importlib.import_module("onnx")
    model = onnx.load(str(path))
    value = json.dumps(config, indent=4)
    for prop in model.metadata_props:
        if prop.key == CONFIG_KEY:
            prop.value = value
            break
    else:
        prop = model.metadata_props.add()
        prop.key = CONFIG_KEY
        prop.value = value
    onnx.save(model, str(path))


def _load_torchscript_config(path: Path) -> dict:
    extra_files = {CONFIG_KEY: ""}
    torch.jit.load(str(path), map_location="cpu", _extra_files=extra_files)
    return json.loads(extra_files[CONFIG_KEY])


def _update_torchscript_config(path: Path, config: dict) -> None:
    module = torch.jit.load(str(path), map_location="cpu")
    torch.jit.save(
        module,
        str(path),
        _extra_files={CONFIG_KEY: json.dumps(config, indent=4)},
    )
