import json
import warnings
from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from torch.utils.data import DataLoader

from rtnls_inference.datasets.fundus import (
    FundusTestDataset,
)
from rtnls_inference.ensembles.onnx_backend import OnnxEnsembleBackend
from rtnls_inference.ensembles.predict_output import (
    PredictFullOutput,
    canonical_context,
    decollate_predict_full,
    to_numpy,
    validate_predict_full,
)
from rtnls_inference.readers import make_mask_reader
from rtnls_inference.release_config import update_stored_config
from rtnls_inference.transforms import make_test_transform
from rtnls_inference.utils import collate_except_metadata


class Ensemble(L.LightningModule):
    predict_full_model = PredictFullOutput
    autocast_inference = False

    def __init__(
        self, ensemble: L.LightningModule, config: dict, fpath: Path | str = None
    ):
        super().__init__()
        self.ensemble = ensemble
        self.config = config
        self.fpath = fpath

    @classmethod
    def from_torchscript(
        cls,
        fpath: str | Path,
        inference_overrides: dict | None = None,
        **kwargs,
    ):
        """Load a TorchScript ensemble, optionally overriding inference config."""
        extra_files = {"config.yaml": ""}  # values will be replaced with data

        ensemble = torch.jit.load(fpath, _extra_files=extra_files).eval()

        config = deepcopy(json.loads(extra_files["config.yaml"]))
        if inference_overrides is not None:
            config.setdefault("inference", {}).update(inference_overrides)

        return cls(ensemble, config, fpath, **kwargs)

    @classmethod
    def from_onnx(
        cls,
        fpath: str | Path,
        inference_overrides: dict | None = None,
        providers: list[str] | None = None,
        **kwargs,
    ):
        """Load an ONNX ensemble release, optionally overriding inference config."""
        fpath = Path(fpath)
        ensemble = OnnxEnsembleBackend(fpath, providers=providers).eval()
        config = deepcopy(ensemble.config)
        if inference_overrides is not None:
            config.setdefault("inference", {}).update(inference_overrides)
        return cls(ensemble, config, fpath, **kwargs)

    @classmethod
    def from_huggingface(cls, modelstr: str, **kwargs):
        repo_name, repo_fpath = modelstr.split(":")
        fpath = hf_hub_download(repo_id=repo_name, filename=repo_fpath)
        if str(fpath).endswith(".onnx"):
            return cls.from_onnx(fpath, **kwargs)
        return cls.from_torchscript(fpath, **kwargs)

    @classmethod
    def from_release(cls, release_name: str, prefer: str | None = None, **kwargs):
        """Load a release from RTNLS_MODEL_RELEASES by name."""
        import os

        from rtnls_inference.ensembles import make_ensemble

        release_path = os.path.join(os.environ["RTNLS_MODEL_RELEASES"], release_name)
        return make_ensemble(release_path, prefer=prefer, **kwargs)

    @classmethod
    def from_modelstring(cls, modelstr: str, **kwargs):
        if modelstr.startswith("hf@"):
            return cls.from_huggingface(modelstr[3:], **kwargs)
        else:
            return cls.from_release(modelstr, **kwargs)

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        if hasattr(self.ensemble, "set_inference_device"):
            self.ensemble.set_inference_device(self.get_device())
        return out

    def get_device(self):
        if hasattr(self.ensemble, "inference_device"):
            return self.ensemble.inference_device
        if next(self.parameters(), None) is not None:
            return next(self.parameters()).device
        return torch.device("cpu")

    @staticmethod
    def _require_batch(batch: Mapping[str, Any]) -> None:
        if not isinstance(batch, Mapping):
            raise TypeError("predict_step requires a mapping containing 'image'")
        if "image" not in batch:
            raise KeyError("predict_step requires batch['image']")
        if not isinstance(batch["image"], torch.Tensor):
            raise TypeError("batch['image'] must be a torch.Tensor")
        if batch["image"].ndim != 4:
            raise ValueError(f"batch['image'] must be NCHW, got {batch['image'].shape}")

    def _model_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Move model input tensors only; leave IDs and canonical context on CPU."""
        self._require_batch(batch)
        model_batch = dict(batch)
        model_batch["image"] = batch["image"].to(self.get_device())
        return model_batch

    def _autocast_context(self):
        device = self.get_device()
        if self.autocast_inference and device.type == "cuda":
            return torch.autocast(device_type="cuda")
        return nullcontext()

    def _predict_member_tensors(self, batch: Mapping[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def _aggregate_tensors(self, member_output: torch.Tensor) -> torch.Tensor:
        return member_output.mean(dim=1)

    def _predict_step_tensors(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        prediction = self._predict_member_tensors(batch)
        aggregate = self._aggregate_tensors(prediction)
        return {"prediction": prediction, "aggregate": aggregate}

    def _prediction_size(
        self, batch: Mapping[str, Any], output: Mapping[str, Any]
    ) -> tuple[int, int] | None:
        return None

    @torch.no_grad()
    def predict_step(
        self, batch: Mapping[str, Any], batch_idx: int | None = None
    ) -> torch.Tensor:
        """Return the member-aggregated tensor-safe prediction on its device."""
        del batch_idx
        model_batch = self._model_batch(batch)
        with self._autocast_context():
            output = self._predict_step_tensors(model_batch)
        aggregate = output["aggregate"]
        if not isinstance(aggregate, torch.Tensor):
            raise TypeError("_predict_step_tensors()['aggregate'] must be a tensor")
        return aggregate.detach()

    @torch.no_grad()
    def predict_step_full(
        self, batch: Mapping[str, Any], batch_idx: int | None = None
    ) -> dict[str, Any]:
        """Return validated batch-major NumPy predictions and inspection context."""
        del batch_idx
        self._require_batch(batch)
        model_batch = self._model_batch(batch)
        with self._autocast_context():
            tensor_output = self._predict_step_tensors(model_batch)
        output = to_numpy(tensor_output)
        batch_size = int(np.asarray(output["aggregate"]).shape[0])
        output.update(
            canonical_context(
                batch,
                batch_size=batch_size,
                prediction_size=self._prediction_size(model_batch, tensor_output),
            )
        )
        return validate_predict_full(self.predict_full_model, output)

    def postprocess_item(self, item: Mapping[str, Any]) -> dict[str, Any]:
        """Convert one full-output item to its canonical public representation."""
        result = {
            "id": item.get("id"),
            "output": np.asarray(item["aggregate"]),
            "output_kind": "prediction",
            "output_space": "preprocessed",
            "geometry": item["geometry"],
        }
        if "preprocessed_image" in item:
            result["preprocessed_image"] = item["preprocessed_image"]
        return result

    def _compatibility_items(
        self, full_output: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        return [
            {"id": item.get("id"), "prediction": item["aggregate"]}
            for item in decollate_predict_full(full_output)
        ]

    def _predict_batch(self, batch: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Deprecated prediction-geometry adapter for legacy integrations."""
        warnings.warn(
            "_predict_batch is deprecated; use predict_step_full and decollate the "
            "batch-major NumPy output",
            FutureWarning,
            stacklevel=2,
        )
        return self._compatibility_items(self.predict_step_full(batch))

    def save_stored_config(
        self,
        path: str | Path | None = None,
        *,
        out_path: str | Path | None = None,
    ) -> Path:
        """Persist ``self.config`` into the release file (.pt or .onnx)."""
        release_path = Path(path or self.fpath)
        if release_path.suffix.lower() not in {".pt", ".onnx"}:
            raise ValueError(
                f"Cannot save config: release path must be .pt or .onnx, got {release_path}"
            )
        return update_stored_config(release_path, self.config, out_path=out_path)

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

    def predict_batch(self, batch):
        pass
