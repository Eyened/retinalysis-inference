from pathlib import Path
from typing import Union

import numpy as np
import pydicom
import torch
from PIL import Image


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
        else:
            return batch

    # Decollate the batch
    decollated = [recursive_decollate(batch, i, None) for i in range(batch_size)]

    # attach the metadata
    for i, item in enumerate(decollated):
        item["metadata"] = metadata[i]

    return decollated


def extract_keypoints_from_heatmaps(heatmaps):
    """Input shape: NMCHW (n_models, batch_size, num_keypoints, height, width)
    Output shape: NMC2
    """
    batch_size, n_models, num_keypoints, _, _ = heatmaps.shape
    outputs = torch.zeros(batch_size, n_models, num_keypoints, 2, dtype=torch.float32)

    for b in range(batch_size):
        for m in range(n_models):
            for i in range(num_keypoints):
                heatmap = heatmaps[b, m, i]
                max_idx = torch.argmax(heatmap)

                n_cols = heatmap.shape[1]
                row = max_idx // n_cols
                col = max_idx % n_cols

                outputs[b, m, i] = torch.tensor([col.item() + 0.5, row.item() + 0.5])
    return outputs


def load_image_pil(path: Union[Path, str]):
    if isinstance(path, str):
        path = Path(path)
    if path.suffix == ".dcm":
        ds = pydicom.dcmread(str(path))
        img = Image.fromarray(ds.pixel_array)
    else:
        img = Image.open(str(path))
    return img


def load_image(path: Union[Path, str], dtype: Union[np.uint8, np.float32] = np.uint8):
    if Path(path).suffix == ".npy":
        im = np.load(path)
    else:
        im = np.array(load_image_pil(path), dtype=np.uint8)
    if im.dtype == np.uint8 and dtype == np.float32:
        im = (im / 255).astype(np.float32)
    if im.dtype == np.float32 and dtype == np.uint8:
        im = np.round(im * 255).astype(np.uint8)
    return im


def find_release_file(release_path: str | Path) -> Path:
    if not isinstance(release_path, Path):
        release_path = Path(release_path)

    assert not bool(release_path.suffix), "release_path should not have a suffix"

    if release_path.with_suffix(".pt").exists():
        return release_path.with_suffix(".pt")
    else:
        raise ValueError(f"No release file found for relase path {release_path}")
