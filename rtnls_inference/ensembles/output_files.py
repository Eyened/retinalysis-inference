"""Small file helpers for table summaries and spatial prediction exports."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from rtnls_inference.datasets.fundus import FundusInferenceDataset

MANIFEST_COLUMNS = [
    "id",
    "output_path",
    "output_kind",
    "output_space",
    "preprocessed_height",
    "preprocessed_width",
    "prediction_height",
    "prediction_width",
]


def prepare_export(dataloader, dest_path, filename, *, overwrite=False):
    """Check known input IDs and existing summaries before running inference."""
    dataset = getattr(dataloader, "dataset", None)
    if isinstance(dataset, FundusInferenceDataset):
        dataset.validate_ids(for_output=True)
    path = Path(dest_path).expanduser().resolve() / filename
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    return path


def output_suffix(item):
    output = np.asarray(item["output"])
    if (
        item["output_kind"] in {"mask", "artery_vein_mask"}
        and output.ndim == 2
        and output.dtype == np.uint8
    ):
        return ".png"
    return ".npy"


def save_output_item(item, path):
    output = np.asarray(item["output"])
    if path.suffix == ".png":
        Image.fromarray(output).save(path)
    else:
        np.save(path, output, allow_pickle=False)


def save_summary(table, path):
    """Serialize list-valued classification cells without changing the returned table."""
    serialized = table.copy()
    for column in serialized.select_dtypes(include="object"):
        serialized[column] = serialized[column].map(
            lambda value: (
                json.dumps(value) if isinstance(value, (list, tuple)) else value
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized.to_csv(path, index=False)
