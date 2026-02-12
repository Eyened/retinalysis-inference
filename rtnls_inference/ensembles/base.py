import json
from pathlib import Path

import lightning as L
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from torch.utils.data import DataLoader

from rtnls_inference.datasets.fundus import (
    FundusTestDataset,
)
from rtnls_inference.readers import make_mask_reader
from rtnls_inference.transforms import make_test_transform
from rtnls_inference.utils import collate_except_metadata


class Ensemble(L.LightningModule):
    def __init__(
        self, ensemble: L.LightningModule, config: dict, fpath: Path | str = None
    ):
        super().__init__()
        self.ensemble = ensemble
        self.config = config
        self.fpath = fpath

    @classmethod
    def from_torchscript(cls, fpath: str | Path, **kwargs):
        extra_files = {"config.yaml": ""}  # values will be replaced with data

        ensemble = torch.jit.load(fpath, _extra_files=extra_files).eval()

        config = json.loads(extra_files["config.yaml"])
        return cls(ensemble, config, fpath, **kwargs)

    @classmethod
    def from_huggingface(cls, modelstr: str, **kwargs):
        repo_name, repo_fpath = modelstr.split(":")
        fpath = hf_hub_download(repo_id=repo_name, filename=repo_fpath)
        return cls.from_torchscript(fpath, **kwargs)

    @classmethod
    def from_modelstring(cls, modelstr: str, **kwargs):
        if modelstr.startswith("hf@"):
            return cls.from_huggingface(modelstr[3:], **kwargs)
        else:
            return cls.from_release(modelstr, **kwargs)

    def hf_upload(self):
        """Upload self.fpath to huggingface"""
        api = HfApi()
        assert "huggingface" in self.config, (
            "config must have a huggingface key with huggingface details."
        )
        fpath = self.fpath
        if not Path(fpath).suffix:
            fpath += ".pt"
        repo_id = self.config["huggingface"]["repo"]
        repo_path = (
            self.config["huggingface"]["path"] + "/" + self.config["name"] + ".pt"
        )
        print(f"Uploading file {fpath} to huggingface: {repo_id}:{repo_path}")
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
        )


class FundusEnsemble(Ensemble):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _make_inference_dataloader(
        self,
        inputs: dict,
        preprocess=True,
        batch_size=None,
        num_workers=8,
        ignore_exceptions=True,
    ):
        datamodule_config = self.config.get("datamodule", {})

        mask_reader = None
        if "mask_reader" in datamodule_config and isinstance(
            datamodule_config["mask_reader"], dict
        ):
            mask_reader = make_mask_reader(datamodule_config["mask_reader"])

        input_mask_reader = None
        if "input_mask_reader" in datamodule_config and isinstance(
            datamodule_config["input_mask_reader"], dict
        ):
            input_mask_reader = make_mask_reader(datamodule_config["input_mask_reader"])

        dataset = FundusTestDataset(
            data=inputs,
            transform=make_test_transform(
                self.config["datamodule"].get("test_transform", {}),
                preprocess=preprocess,
            ),
            ignore_exceptions=ignore_exceptions,
            mask_reader=mask_reader,
            input_mask_reader=input_mask_reader,
        )

        batch_size = (
            batch_size
            if batch_size is not None
            else self.config["inference"].get("batch_size", 8)
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=False,
            shuffle=False,
            collate_fn=(
                collate_except_metadata
                if ignore_exceptions
                else torch.utils.data.dataloader.default_collate
            ),
            num_workers=num_workers,
        )

    def predict_dataset(
        self,
        data,
        dest_path=None,
        num_workers=0,
        batch_size=None,
    ):
        """Run inference on a dataset.

        Args:
            data: List of dicts, each containing 'id', 'image', and optionally 'contrast_enhanced'
            dest_path: Directory to save predictions
            num_workers: Number of dataloader workers
            batch_size: Batch size for inference
        """
        inputs = {"images": data}
        dataloader = self._make_inference_dataloader(
            inputs,
            num_workers=num_workers,
            preprocess=True,
            batch_size=batch_size,
        )
        return self._predict_dataloader(dataloader, dest_path)

    def predict(self, *args, **kwargs):
        """Alias for predict_dataset to maintain backward compatibility."""
        return self.predict_dataset(*args, **kwargs)

    def predict_preprocessed(
        self,
        data,
        dest_path=None,
        num_workers=0,
        batch_size=None,
    ):
        """Run inference on preprocessed images.

        Args:
            data: List of dicts, each containing 'id', 'image', and optionally 'contrast_enhanced'
            dest_path: Directory to save predictions
            num_workers: Number of dataloader workers
            batch_size: Batch size for inference
        """
        inputs = {"images": data}
        dataloader = self._make_inference_dataloader(
            inputs,
            num_workers=num_workers,
            preprocess=False,
            batch_size=batch_size,
        )
        return self._predict_dataloader(dataloader, dest_path)

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        dest_path=None,
        image_path_column="image",
        mask_path_column="mask",
        id_column="id",
        preprocess=True,
        **kwargs,
    ):
        """Run inference on a pandas DataFrame.

        Args:
            df: DataFrame with image paths
            dest_path: Directory to save predictions
            image_path_column: Column name for image paths
            mask_path_column: Column name for mask paths (optional)
            id_column: Column name for IDs (optional, will use index if not present)
            preprocess: Whether to preprocess images
            **kwargs: Additional arguments passed to predict methods
        """
        data = []
        for idx, row in df.iterrows():
            entry = {
                "id": row.get(id_column, str(idx))
                if id_column in df.columns
                else str(idx),
                "image": str(row[image_path_column]),
            }
            if mask_path_column in df.columns:
                entry["mask"] = str(row[mask_path_column])
            data.append(entry)

        if preprocess:
            return self.predict(data, dest_path, **kwargs)
        else:
            return self.predict_preprocessed(data, dest_path, **kwargs)

    def get_device(self):
        # Check if the module has any parameters
        if next(self.parameters(), None) is not None:
            # Return the device of the first parameter
            return next(self.parameters()).device
        else:
            # Fallback or default device if the module has no parameters
            # This might be necessary for modules that do not have parameters
            # and hence might not have a clear device assignment.
            # Adjust this part based on your specific needs.
            return torch.device("cpu")

    def predict_batch(self, batch):
        pass
