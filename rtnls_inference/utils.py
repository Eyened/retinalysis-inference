from pathlib import Path
from typing import Any, Optional, Sequence, Union

import io
import numpy as np
import torch
from PIL import Image

try:
    import simplejpeg  # type: ignore
except Exception:  # pragma: no cover
    simplejpeg = None


def _normalize_suffix(suffix: Optional[str]) -> str:
    if not suffix:
        return ""
    if not suffix.startswith("."):
        suffix = "." + suffix
    return suffix.lower()


def _decode_image_pil(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        return np.array(im)


def _decode_jpeg_simplejpeg(data: bytes) -> np.ndarray:
    if simplejpeg is None:
        raise RuntimeError("simplejpeg is not available")
    return simplejpeg.decode_jpeg(data, colorspace="RGB")


def test_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.tensor([])  # Return an empty tensor if all items are faulty
    # Convert batch list to a tensor or any suitable format for your model
    # This depends on the structure of your data items
    return torch.utils.data.dataloader.default_collate(batch)


def get_all_subclasses_dict(cls):
    all_subclasses_dict = {}

    for subclass in cls.__subclasses__():
        all_subclasses_dict[subclass.__name__] = subclass
        all_subclasses_dict.update(get_all_subclasses_dict(subclass))

    return all_subclasses_dict


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def remove_empty_lists(d):
    if isinstance(d, dict):
        return {
            k: remove_empty_lists(v)
            for k, v in d.items()
            if not (isinstance(v, (list, torch.Tensor)) and len(v) == 0)
        }
    else:
        return d


def collate_except_metadata(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        raise ValueError("Batch is empty")
    collated = {}
    for key in batch[0].keys():
        if key == "metadata":
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = torch.utils.data.default_collate(
                [item[key] for item in batch]
            )
    return collated


def decollate_batch(batch):
    """
    Separate batched PyTorch tensors in a nested dictionary into individual items and convert them to numpy or primitive types if the size is 1.

    Args:
        batch (dict): A dictionary where each key has a tensor value batched along the first dimension, lists, or nested dictionaries.

    Returns:
        list: A list of dictionaries, where each dictionary represents an item from the original batch.
    """
    # Number of items in the batch, assuming all tensors have the same batch size

    assert "id" in batch, "Batch must contain an 'id' key to use decollate_batch"
    batch_size = len(batch["id"])

    # remove batch elements with zero length
    batch = remove_empty_lists(batch)

    if "metadata" in batch:
        metadata = batch["metadata"]
        del batch["metadata"]
    else:
        metadata = [{} for _ in range(batch_size)]

    def convert(val):
        if isinstance(val, torch.Tensor):
            decollated_val = val.detach().cpu().numpy()
            if decollated_val.size == 1:
                return decollated_val.item()
            return decollated_val
        elif isinstance(val, dict):
            return decollate_batch(val)
        elif isinstance(val, np.ndarray):
            if val.size == 1:
                return val.item()
            return val
        elif isinstance(val, list):
            return [convert(item) for item in val]
        else:
            return val

    # Recursive function to decollate nested dictionaries and lists
    def recursive_decollate(batch, index, key):
        if isinstance(batch, dict):
            return {
                key: recursive_decollate(value, index, key)
                for key, value in batch.items()
            }
        elif isinstance(batch, list):
            return convert(batch[index])
        elif isinstance(batch, torch.Tensor):
            return convert(batch[index])
        elif isinstance(batch, np.ndarray):
            return convert(batch[index])
        else:
            return batch

    # Decollate the batch
    decollated = [recursive_decollate(batch, i, None) for i in range(batch_size)]

    # attach the metadata
    for i, item in enumerate(decollated):
        item["metadata"] = metadata[i]

    return decollated


def format_keypoints_for_transform(
    keypoints: Any,
    keypoint_names: Optional[Sequence[str]] = None,
) -> list:
    """Convert named dataset keypoints to Albumentations' ordered xy list."""
    if keypoints is None:
        return []

    if isinstance(keypoints, dict):
        names = keypoint_names or sorted(keypoints)
        missing = [name for name in names if name not in keypoints]
        if missing:
            raise KeyError(f"Missing keypoints for configured names: {missing}")

        formatted = []
        for name in names:
            coords = keypoints[name]
            if len(coords) != 2:
                raise ValueError(
                    f"keypoints['{name}'] must contain exactly [x, y], got {coords}"
                )
            formatted.append([float(coords[0]), float(coords[1])])
        return formatted

    return keypoints


def format_keypoints_to_tensor(keypoints: Any) -> torch.Tensor:
    """Convert transformed keypoints to the tensor shape expected by LMs."""
    if torch.is_tensor(keypoints):
        return keypoints.to(torch.float32)
    return torch.as_tensor(keypoints, dtype=torch.float32)


def extract_keypoints_from_heatmaps(heatmaps):
    """Input shape: NMKHW (batch, members, keypoints, height, width).

    Output shape: NMK2.
    """
    if heatmaps.ndim != 5:
        raise ValueError(f"Expected NMKHW heatmaps, got {heatmaps.shape}")
    width = heatmaps.shape[-1]
    indices = heatmaps.flatten(start_dim=-2).argmax(dim=-1)
    rows = torch.div(indices, width, rounding_mode="floor")
    columns = indices.remainder(width)
    return torch.stack((columns, rows), dim=-1).to(torch.float32) + 0.5


def load_image_pil(path: Union[Path, str]) -> Image.Image:
    if isinstance(path, str):
        path = Path(path)
    if path.suffix == ".dcm":
        raise ValueError("DICOM images (.dcm) are no longer supported")
    return Image.open(str(path))


def read_file_bytes(path: Union[Path, str]) -> bytes:
    """Read a file's raw bytes."""
    if isinstance(path, str):
        path = Path(path)
    return path.read_bytes()


def load_image_from_bytes(
    data: bytes,
    dtype: Union[np.uint8, np.float32] = np.uint8,
    suffix: Optional[str] = None,
) -> np.ndarray:
    """Decode an encoded image from bytes into an RGB numpy array."""
    suffix = _normalize_suffix(suffix)
    if suffix in (".jpg", ".jpeg"):
        try:
            arr = _decode_jpeg_simplejpeg(data)
        except Exception:
            arr = _decode_image_pil(data)
    else:
        arr = _decode_image_pil(data)

    if dtype == np.float32:
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / 255.0
    elif dtype == np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255)

    return arr.astype(dtype)


def load_image(path: Union[Path, str], dtype: Union[np.uint8, np.float32] = np.uint8):
    if isinstance(path, str):
        path = Path(path)
    data = read_file_bytes(path)
    return load_image_from_bytes(data=data, dtype=dtype, suffix=path.suffix)


def find_release_file(
    release_path: str | Path,
    prefer: str | None = None,
) -> Path:
    """Resolve a release path to an existing .pt or .onnx file.

    Supports legacy flat files (`<name>.pt`) and foldered releases
    (`<name>/<name>.pt`).
    """
    if not isinstance(release_path, Path):
        release_path = Path(release_path)

    assert not release_path.suffix, "release_path should not have a suffix"

    pt_path = release_path.with_suffix(".pt")
    onnx_path = release_path.with_suffix(".onnx")
    nested_pt = release_path / f"{release_path.name}.pt"
    nested_onnx = release_path / f"{release_path.name}.onnx"
    if prefer == "onnx":
        candidates = [onnx_path, nested_onnx, pt_path, nested_pt]
    else:
        candidates = [pt_path, nested_pt, onnx_path, nested_onnx]

    for path in candidates:
        if path.exists():
            return path

    tried = ", ".join(str(path) for path in candidates)
    raise ValueError(
        f"No release file found for release path {release_path} (tried {tried})"
    )


from rtnls_inference.release_config import (  # noqa: E402
    load_stored_config,
    update_stored_config,
)
