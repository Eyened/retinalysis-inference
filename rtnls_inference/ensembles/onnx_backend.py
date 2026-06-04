import importlib
from pathlib import Path

import torch
import torch.nn as nn

from rtnls_inference.release_config import load_stored_config


def load_config_from_onnx(path: str | Path) -> dict:
    """Load embedded training config from ONNX metadata."""
    return load_stored_config(path)


class OnnxEnsembleBackend(nn.Module):
    """Torch module wrapper around an ONNX Runtime session for ensemble forward passes."""

    def __init__(
        self,
        onnx_path: str | Path,
        providers: list[str] | None = None,
    ):
        super().__init__()
        try:
            ort = importlib.import_module("onnxruntime")
        except ImportError as exc:
            raise ImportError(
                "ONNX inference requires `onnxruntime`. "
                "Install retinalysis-inference with the `onnx` extra."
            ) from exc

        self.onnx_path = Path(onnx_path)
        self.config = load_config_from_onnx(self.onnx_path)
        self.input_name = "image"
        self.output_name = "prediction"
        self._providers_override = providers
        self.register_buffer("_device_anchor", torch.zeros(1))
        self._session = self._make_session(self._device_anchor.device)

    @property
    def inference_device(self) -> torch.device:
        return self._device_anchor.device

    def set_inference_device(self, device: torch.device) -> None:
        """Recreate the ORT session when the inference device type changes."""
        if device.type == self._device_anchor.device.type:
            return
        self._device_anchor = self._device_anchor.to(device)
        self._session = self._make_session(device)

    def _make_session(self, device: torch.device) -> object:
        ort = importlib.import_module("onnxruntime")
        if self._providers_override is not None:
            providers = self._providers_override
        elif device.type == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        return ort.InferenceSession(str(self.onnx_path), providers=providers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().float().cpu().numpy()
        out = self._session.run(
            [self.output_name],
            {self.input_name: x_np},
        )[0]
        return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)
