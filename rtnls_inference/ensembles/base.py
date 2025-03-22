import json
import os
from pathlib import Path

import lightning as L
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from torch.utils.data import DataLoader

from rtnls_inference.datasets.fundus import (
    FundusTestDataset,
)
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
    def from_release(cls, fname: str, **kwargs):
        if os.path.exists(fname):
            fpath = fname
        else:
            fpath = os.path.join(os.environ["RTNLS_MODEL_RELEASES"], fname)

        fpath = Path(fpath)
        if fpath.suffix == ".pt":
            return cls.from_torchscript(fpath, **kwargs)
        else:
            raise ValueError(f"Unrecognized extension {fpath.suffix}")

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

    def _make_test_dataloader(
        self, dataframe_path: str | Path, base_path: str | Path = None
    ):
        from rtnls_models.data_loading.dm_dataframe import DataframeDataModule

        dm = DataframeDataModule.from_path(dataframe_path, base_path)
        dm.setup()
        return dm.test_dataloader()

    def _make_inference_dataloader(
        self,
        image_paths,
        bounds=None,
        ids=None,
        preprocess=True,
        batch_size=None,
        num_workers=8,
        ignore_exceptions=True,
    ):
        contrast_enhance = (
            (
                isinstance(image_paths[0], str)
                or isinstance(image_paths[0], Path)
                or (len(image_paths[0]) == 1)
            )
            if self.config["datamodule"]["test_transform"].get("contrast_enhance", True)
            else False
        )

        dataset = FundusTestDataset(
            images_paths=image_paths,
            bounds=bounds,
            ids=ids,
            transform=make_test_transform(
                self.config,
                preprocess=preprocess,
                contrast_enhance=contrast_enhance,
            ),
            ignore_exceptions=True,
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

    def predict(
        self,
        image_paths,
        bounds=None,
        ids=None,
        dest_path=None,
        num_workers=0,
        batch_size=None,
    ):
        dataloader = self._make_inference_dataloader(
            image_paths,
            bounds=bounds,
            ids=ids,
            num_workers=num_workers,
            preprocess=True,
            batch_size=batch_size,
        )
        return self._predict_dataloader(dataloader, dest_path)

    def predict_preprocessed(
        self,
        image_paths,
        ids=None,
        dest_path=None,
        num_workers=0,
        batch_size=None,
    ):
        dataloader = self._make_inference_dataloader(
            image_paths,
            ids=ids,
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
        preprocess=True,
        **kwargs,
    ):
        image_paths = df[image_path_column].to_list()
        if preprocess:
            return self.predict(image_paths, dest_path, **kwargs)
        else:
            return self.predict_preprocessed(image_paths, dest_path, **kwargs)

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
