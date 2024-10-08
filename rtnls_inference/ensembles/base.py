import json
import os
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rtnls_inference.datasets.fundus import (
    FundusTestDataset,
    normalizer,
    to_tensor,
)
from rtnls_inference.transforms import make_test_transform
from rtnls_inference.utils import decollate_batch, test_collate_fn
from huggingface_hub import HfApi, hf_hub_download


class Ensemble(L.LightningModule):
    def __init__(
        self,
        ensemble: L.LightningModule,
        config: dict,
        fpath: Path | str = None
    ):
        super().__init__()
        self.ensemble = ensemble
        self.config = config
        self.fpath = fpath

    @classmethod
    def from_torchscript(cls, fpath: str | Path):
        extra_files = {"config.yaml": ""}  # values will be replaced with data

        ensemble = torch.jit.load(fpath, _extra_files=extra_files).eval()

        config = json.loads(extra_files["config.yaml"])
        return cls(ensemble, config, fpath)

    @classmethod
    def from_release(cls, fname: str):
        if os.path.exists(fname):
            fpath = fname
        else:
            fpath = os.path.join(os.environ["RTNLS_MODEL_RELEASES"], fname)

        fpath = Path(fpath)
        if fpath.suffix == ".pt":
            return cls.from_torchscript(fpath)
        else:
            raise ValueError(f"Unrecognized extension {fpath.suffix}")
        
    @classmethod
    def from_huggingface(cls, modelstr: str):
        repo_name, repo_fpath = modelstr.split(':')
        fpath = hf_hub_download(repo_id=repo_name, filename=repo_fpath)
        return cls.from_torchscript(fpath)
    
    def hf_upload(self):
        ''' Upload self.fpath to huggingface
        '''
        api = HfApi()
        assert 'huggingface' in self.config, 'config must have a huggingface key with huggingface details.'
        fpath = self.fpath
        if not Path(fpath).suffix:
            fpath += '.pt'
        repo_id = self.config['huggingface']['repo']
        repo_path = self.config['huggingface']['path'] + '/' + self.config['name'] + '.pt'
        print(f'Uploading file {fpath} to huggingface: {repo_id}:{repo_path}')
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
        )



class FundusEnsemble(Ensemble):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = make_test_transform(self.config)

    def make_batch(self, images, preprocess=False):
        batch = []
        for image in images:
            item = {"image": image, "keypoints": []}
            item = self.transform(**item, preprocess=preprocess)
            item = normalizer(**item)
            if "ce" in item:
                item["image"] = np.concatenate([item["image"], item.pop("ce")], axis=-1)
            item = to_tensor(**item)
            batch.append(item)

        return test_collate_fn(batch)

    def _make_dataloader(
        self,
        image_paths,
        preprocess,
        batch_size=None,
        num_workers=8,
        ignore_exceptions=True,
    ):
        contrast_enhance = (
            isinstance(image_paths[0], str)
            or isinstance(image_paths[0], Path)
            or (len(image_paths[0]) == 1)
        )
        dataset = FundusTestDataset(
            images_paths=image_paths,
            transform=make_test_transform(
                self.config,
                preprocess=preprocess,
                contrast_enhance=contrast_enhance,
            ),
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
                test_collate_fn
                if ignore_exceptions
                else torch.utils.data.dataloader.default_collate
            ),
            num_workers=num_workers,
        )

    def predict(
        self,
        image_paths,
        dest_path=None,
        num_workers=0,
        batch_size=None,
    ):
        dataloader = self._make_dataloader(
            image_paths, num_workers=num_workers, preprocess=True, batch_size=batch_size
        )
        return self._predict_dataloader(dataloader, dest_path)

    def predict_preprocessed(
        self,
        image_paths,
        dest_path=None,
        num_workers=0,
        batch_size=None,
    ):
        dataloader = self._make_dataloader(
            image_paths,
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

    def _predict_batch(self, batch):
        items = self.predict_batch(batch)
        if "bounds" in items:
            items["bounds"] = batch["bounds"]
        items = decollate_batch(items)
        items = [self.transform.undo_item(item) for item in items]
        return items

    def predict_images(self, images, preprocess=False):
        """Input: list of numpy images of potentially different shapes"""
        batch = self.make_batch(images, preprocess=preprocess)
        items = self._predict_batch(batch)

        return items
