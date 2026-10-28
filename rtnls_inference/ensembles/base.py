import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, TypeVar

import lightning as L
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rtnls_inference.datasets.fundus import FundusInferenceDataset
from rtnls_inference.ensembles.onnx_backend import OnnxEnsembleBackend
from rtnls_inference.ensembles.output_files import prepare_export, save_summary
from rtnls_inference.ensembles.predict_output import (
    check_prediction_tensors,
    unbatch_predictions,
)
from rtnls_inference.inputs import ImageInput, batch_ids, validate_ids
from rtnls_inference.readers import make_mask_reader
from rtnls_inference.release_config import update_stored_config
from rtnls_inference.transforms import make_test_transform
from rtnls_inference.utils import collate_inference_items

T = TypeVar("T")
_UNSET: Any = object()

_INFERENCE_OVERRIDE_NAMES = (
    "batch_size",
    "tile_batch_size",
    "tta",
    "tta_flips",
    "tracing_input_size",
    "overlap",
    "blend",
    "gaussian_sigma_scale",
    "return_postprocess_intermediates",
    "graph_refinement",
)


class Ensemble(L.LightningModule):
    autocast_inference = False
    # Fields retained on the inference device for postprocess_item. Inspection
    # fields (member predictions, heatmaps, etc.) need not cross to the CPU.
    postprocess_fields = ("aggregate",)

    def __init__(
        self,
        ensemble: L.LightningModule,
        config: dict,
        fpath: Path | str | None = None,
        *,
        batch_size: int | None = _UNSET,
        tile_batch_size: int | None = _UNSET,
        tta: bool | None = _UNSET,
        tta_flips: Sequence[Sequence[int]] | None = _UNSET,
        tracing_input_size: Sequence[int] | None = _UNSET,
        overlap: float | None = _UNSET,
        blend: str | None = _UNSET,
        gaussian_sigma_scale: float | None = _UNSET,
        return_postprocess_intermediates: bool | None = _UNSET,
        graph_refinement: Mapping[str, Any] | None = _UNSET,
    ):
        super().__init__()
        self.ensemble = ensemble
        self.config = config
        self.fpath = fpath
        supplied = {
            "batch_size": batch_size,
            "tile_batch_size": tile_batch_size,
            "tta": tta,
            "tta_flips": tta_flips,
            "tracing_input_size": tracing_input_size,
            "overlap": overlap,
            "blend": blend,
            "gaussian_sigma_scale": gaussian_sigma_scale,
            "return_postprocess_intermediates": return_postprocess_intermediates,
            "graph_refinement": graph_refinement,
        }
        self._inference_overrides = {
            name: value for name, value in supplied.items() if value is not _UNSET
        }

    def _inference_config(self) -> dict[str, Any]:
        inference = self.config.get("inference")
        return inference if isinstance(inference, Mapping) else {}

    def _inference_setting(self, name: str, default: T) -> T:
        """Resolve a runtime inference setting: constructor, then config, then default."""
        if name not in _INFERENCE_OVERRIDE_NAMES:
            raise KeyError(f"Unknown inference setting {name!r}")
        if name in self._inference_overrides:
            return self._inference_overrides[name]
        inference = self._inference_config()
        if name in inference:
            return inference[name]
        return default

    def _inference_mapping(
        self, name: str, default: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Merge a nested inference mapping: default, then config, then constructor."""
        if name not in _INFERENCE_OVERRIDE_NAMES:
            raise KeyError(f"Unknown inference setting {name!r}")
        merged: dict[str, Any] = dict(default or {})
        embedded = self._inference_config().get(name)
        if isinstance(embedded, Mapping):
            merged.update(embedded)
        override = self._inference_overrides.get(name, _UNSET)
        if override is not _UNSET and override is not None:
            if not isinstance(override, Mapping):
                raise TypeError(f"{name} must be a mapping, got {type(override)!r}")
            merged.update(override)
        return merged

    def _tile_batch_size(self, default: int, *, legacy_batch_size: bool = False) -> int:
        """Resolve sliding-window tile batch size without using constructor batch_size."""
        override = self._inference_overrides.get("tile_batch_size", _UNSET)
        if override is not _UNSET:
            return int(override)
        inference = self._inference_config()
        if "tile_batch_size" in inference:
            return int(inference["tile_batch_size"])
        if legacy_batch_size and "batch_size" in inference:
            return int(inference["batch_size"])
        return int(default)

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
        try:
            # Trigger HF download counts; runtime settings stay embedded.
            hf_hub_download(repo_id=repo_name, filename="config.yaml")
        except EntryNotFoundError:
            pass  # Older repositories may not include a root config.
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
            raise TypeError(
                f"batch['image'] must be a torch.Tensor, got {type(batch['image']).__name__}"
            )
        if batch["image"].ndim != 4:
            raise ValueError(f"batch['image'] must be NCHW, got {batch['image'].shape}")
        if any(size == 0 for size in batch["image"].shape):
            raise ValueError(
                f"batch['image'] dimensions must be positive, got {tuple(batch['image'].shape)}"
            )
        batch_ids(batch)

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

    @contextmanager
    def _inference_context(self):
        # TorchScript CUDA compilation can use the current device even when
        # model parameters and inputs already reside on another GPU.
        device = self.get_device()
        device_context = (
            torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        )
        with device_context, self._autocast_context():
            yield

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

    def _run_prediction(self, batch):
        model_batch = self._model_batch(batch)
        with self._inference_context():
            output = self._predict_step_tensors(model_batch)
        check_prediction_tensors(output, int(batch["image"].shape[0]))
        return model_batch, output

    @staticmethod
    def make_batch(items: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
        """Collate prepared CHW items; preserve metadata and skip failed inputs.

        Returns an empty mapping when all items were skipped. Model inputs with
        different shapes must be processed in separate batches.
        """
        return collate_inference_items(items)

    @torch.no_grad()
    def predict_batch(
        self,
        batch: Mapping[str, Any],
        *,
        fields: Literal["all"] | Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Predict once, retaining detached tensors on the inference device.

        By default retain only ``postprocess_fields``. A sequence adds named
        inspection fields (for example ``("logits",)``); ``"all"`` retains every
        backend field. ``prediction_size`` is always included. Arrays remain at
        model resolution: use unbatch_predictions for raw NumPy inspection or
        postprocess_batch for final results in canonical preprocessing space.
        """
        if fields is not None and not (isinstance(fields, str) and fields == "all"):
            if isinstance(fields, str) or not isinstance(fields, Sequence):
                raise TypeError(
                    "fields must be None, 'all', or a sequence of field names"
                )
            if any(not isinstance(name, str) for name in fields):
                raise TypeError("fields must contain only string field names")
        model_batch, output = self._run_prediction(batch)
        if isinstance(fields, str):
            names = list(output)
        else:
            names = list(dict.fromkeys((*self.postprocess_fields, *(fields or ()))))
        missing = [
            name for name in names if name not in output and name != "prediction_size"
        ]
        if missing:
            raise ValueError(
                f"Requested prediction fields {missing!r} are unavailable; "
                f"available fields: {sorted((*output, 'prediction_size'))!r}"
            )
        result = {
            name: output[name].detach()
            if isinstance(output[name], torch.Tensor)
            else output[name]
            for name in names
            if name != "prediction_size"
        }
        result["prediction_size"] = self._prediction_size(model_batch, output)
        return result

    def postprocess_batch(
        self, predictions: Mapping[str, Any], batch: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """Convert required fields and finish each prediction in canonical space.

        Inspection-only tensors are not transferred to CPU. This method neither
        runs the model nor writes files; postprocess_item defines family behavior.
        """
        self._require_batch(batch)
        required = (*self.postprocess_fields, "prediction_size")
        missing = [name for name in required if name not in predictions]
        if missing:
            raise ValueError(f"Missing postprocessing fields: {missing!r}")
        retained = {name: predictions[name] for name in required}
        return [
            self.postprocess_item(item) for item in unbatch_predictions(retained, batch)
        ]

    def iter_predictions(self, dataloader):
        """Yield final CPU results one at a time, retaining at most one batch.

        Results include the input context needed for visualization. This method
        never writes files; use it to consume image outputs without collecting
        the entire dataset in memory.
        """
        for batch in tqdm(dataloader):
            if not batch:
                continue
            predictions = self.predict_batch(batch)
            results = self.postprocess_batch(predictions, batch)
            del predictions
            yield from results
            del results, batch

    prediction_prefix = "prediction"

    def _prediction_columns(self, count):
        return [f"{self.prediction_prefix}_{index}" for index in range(count)]

    def _summary_columns(self, count=0):
        count = count or self.config.get("lightningmodule", {}).get("n_class", 0)
        return ["id", *self._prediction_columns(count)]

    def _summary_row(self, item):
        values = np.asarray(item["output"]).reshape(-1)
        return {
            "id": item.get("id"),
            **dict(zip(self._prediction_columns(len(values)), values)),
        }

    def predict_dataloader(
        self,
        dataloader,
        dest_path: str | Path | None = None,
        *,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """Return small final predictions as a table, optionally saving predictions.csv.

        Only numeric predictions and labels are collected; input images and
        inspection intermediates are released as batches are consumed. Spatial
        ensembles override this method to write their outputs incrementally.
        """
        path = (
            prepare_export(
                dataloader, dest_path, "predictions.csv", overwrite=overwrite
            )
            if dest_path is not None
            else None
        )
        rows = []
        columns = None
        for item in self.iter_predictions(dataloader):
            row = self._summary_row(item)
            if columns is None:
                columns = list(row)
            elif list(row) != columns:
                raise ValueError("Prediction columns changed between inputs")
            rows.append(row)
        table = pd.DataFrame(
            rows, columns=columns if columns is not None else self._summary_columns()
        )
        validate_ids(table["id"], for_output=path is not None)
        if path is not None:
            save_summary(table, path)
        return table

    @torch.no_grad()
    def predict_step(
        self, batch: Mapping[str, Any], batch_idx: int | None = None
    ) -> torch.Tensor:
        """Return the member-aggregated tensor-safe prediction on its device."""
        del batch_idx
        _, output = self._run_prediction(batch)
        return output["aggregate"].detach()

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
    inference_dataset_cls = FundusInferenceDataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _preprocess_input_space(input_space: str) -> bool:
        if input_space not in {"raw", "preprocessed"}:
            raise ValueError("input_space must be 'raw' or 'preprocessed'")
        return input_space == "raw"

    def preprocess_item(
        self,
        sample: ImageInput,
        *,
        input_space: Literal["raw", "preprocessed"] = "raw",
    ) -> dict[str, Any]:
        """Load and prepare one input record using the same dataset as bulk runs.

        The image may be a path, RGB uint8 HWC NumPy array, or RGB uint8 CHW
        tensor. IDs default to the filename stem for paths and are otherwise
        optional until exporting. Other input fields remain supported
        (input mask/contrast-enhanced paths, metadata and fov). The result is a
        model-ready CHW CPU tensor, accompanied by canonical RGB and context.
        Already-preprocessed inputs still receive model-specific transforms.
        """
        dataset = self._make_inference_dataset(
            {"images": [sample]},
            preprocess=self._preprocess_input_space(input_space),
            ignore_exceptions=False,
        )
        return dataset[0]

    def make_inference_dataloader(
        self,
        data: Sequence[ImageInput],
        *,
        input_space: Literal["raw", "preprocessed"] = "raw",
        batch_size: int | None = None,
        num_workers: int = 0,
        ignore_exceptions: bool = False,
    ) -> DataLoader:
        """Build a lazy inference loader from input records.

        Input failures raise by default. With ignore_exceptions=True the dataset
        warns and skips loading failures; an entirely skipped batch is empty.
        Subclasses may override this public method to constrain image batching.
        """
        dataset = self._make_inference_dataset(
            {"images": list(data)},
            preprocess=self._preprocess_input_space(input_space),
            ignore_exceptions=ignore_exceptions,
        )
        if batch_size is None:
            batch_size = self._inference_setting("batch_size", 8)
        loader_kwargs = {
            "batch_size": batch_size,
            "pin_memory": self.get_device().type == "cuda",
            "shuffle": False,
            # A static collator keeps the GPU model out of worker processes.
            "collate_fn": type(self).make_batch,
            "num_workers": num_workers,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2
        return DataLoader(dataset, **loader_kwargs)

    def _make_inference_dataset(
        self,
        inputs: dict,
        preprocess=True,
        ignore_exceptions=False,
    ):
        datamodule_config = self.config.get("datamodule", {})

        input_mask_reader = None
        if "input_mask_reader" in datamodule_config and isinstance(
            datamodule_config["input_mask_reader"], dict
        ):
            input_mask_reader = make_mask_reader(datamodule_config["input_mask_reader"])

        return self.inference_dataset_cls(
            data=inputs,
            transform=make_test_transform(
                self.config["datamodule"].get("test_transform", {}),
                preprocess=preprocess,
            ),
            ignore_exceptions=ignore_exceptions,
            input_mask_reader=input_mask_reader,
        )
