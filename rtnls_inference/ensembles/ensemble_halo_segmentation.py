from __future__ import annotations

import torch
import torch.nn.functional as F
from rtnls_inference.ensembles.ensemble_segmentation import SegmentationEnsemble


def _tile_positions(length: int, output_size: int, stride: int) -> list[int]:
    canvas = max(length, output_size)
    positions = list(range(0, max(canvas - output_size, 0) + 1, stride))
    final = canvas - output_size
    if not positions or positions[-1] != final:
        positions.append(final)
    return positions


def gaussian_importance_map(
    output_size: int,
    *,
    sigma_scale: float = 0.125,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    coords = torch.arange(output_size, device=device, dtype=dtype)
    center = (output_size - 1) / 2.0
    sigma = max(output_size * sigma_scale, 1.0)
    one_dimensional = torch.exp(-0.5 * ((coords - center) / sigma) ** 2)
    importance = one_dimensional[:, None] * one_dimensional[None, :]
    return importance.clamp_min(torch.finfo(dtype).eps)


def halo_sliding_window_inference(
    image: torch.Tensor,
    predictor,
    *,
    context_size: int = 1024,
    output_size: int = 512,
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    sigma_scale: float = 0.125,
) -> torch.Tensor:
    """Stitch central halo predictions from a model or model ensemble."""
    if image.ndim != 4:
        raise ValueError(f"Expected NCHW input, got {image.shape}")
    if context_size <= output_size or (context_size - output_size) % 2:
        raise ValueError("context_size/output_size do not define a symmetric halo")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    stride = max(round(output_size * (1.0 - overlap)), 1)
    halo = (context_size - output_size) // 2
    importance = gaussian_importance_map(
        output_size,
        sigma_scale=sigma_scale,
        device=image.device,
        dtype=image.dtype,
    )

    sample_outputs: list[torch.Tensor] = []
    predictor_has_model_axis: bool | None = None
    for sample in image:
        height, width = sample.shape[-2:]
        canvas_h, canvas_w = max(height, output_size), max(width, output_size)
        bottom_extra, right_extra = canvas_h - height, canvas_w - width
        padded = F.pad(
            sample[None],
            (halo, halo + right_extra, halo, halo + bottom_extra),
            mode="constant",
            value=0.0,
        )[0]
        positions = [
            (top, left)
            for top in _tile_positions(canvas_h, output_size, stride)
            for left in _tile_positions(canvas_w, output_size, stride)
        ]

        accumulator = None
        normalizer = torch.zeros(
            (canvas_h, canvas_w), device=image.device, dtype=image.dtype
        )
        for start in range(0, len(positions), sw_batch_size):
            chunk_positions = positions[start : start + sw_batch_size]
            tiles = torch.stack(
                [
                    padded[
                        :,
                        top : top + context_size,
                        left : left + context_size,
                    ]
                    for top, left in chunk_positions
                ]
            )
            predictions = predictor(tiles)
            has_model_axis = predictions.ndim == 5
            if predictions.ndim == 4:
                predictions = predictions[None]
            elif predictions.ndim != 5:
                raise ValueError(
                    f"Halo predictor must return TCHW or MTCHW, got {predictions.shape}"
                )
            if predictor_has_model_axis is None:
                predictor_has_model_axis = has_model_axis
            elif predictor_has_model_axis != has_model_axis:
                raise RuntimeError("Predictor output rank changed between tile batches")
            if predictions.shape[-2:] != (output_size, output_size):
                raise ValueError(
                    f"Halo predictor returned {predictions.shape[-2:]}; "
                    f"expected {(output_size, output_size)}"
                )
            if accumulator is None:
                accumulator = torch.zeros(
                    (
                        predictions.shape[0],
                        predictions.shape[2],
                        canvas_h,
                        canvas_w,
                    ),
                    device=predictions.device,
                    dtype=predictions.dtype,
                )
            for tile_idx, (top, left) in enumerate(chunk_positions):
                accumulator[
                    ..., top : top + output_size, left : left + output_size
                ] += predictions[:, tile_idx] * importance
                normalizer[top : top + output_size, left : left + output_size] += (
                    importance
                )

        if accumulator is None or torch.any(normalizer <= 0):
            raise RuntimeError("Halo tiling left output pixels uncovered")
        sample_outputs.append(
            accumulator[..., :height, :width] / normalizer[:height, :width]
        )

    output = torch.stack(sample_outputs, dim=1)  # M,N,C,H,W
    if predictor_has_model_axis:
        return output.permute(1, 0, 2, 3, 4)  # N,M,C,H,W
    return output[0]  # N,C,H,W


class HaloSegmentationEnsemble(SegmentationEnsemble):
    """Generic softmax segmentation ensemble for center-cropped halo models."""

    def sliding_window_inference(self, image: torch.Tensor) -> torch.Tensor:
        model_config = self.config.get("lightningmodule", {})
        return halo_sliding_window_inference(
            image,
            self.ensemble,
            context_size=int(model_config.get("context_size", 768)),
            output_size=int(model_config.get("output_size", 512)),
            overlap=float(self._inference_setting("overlap", 0.5)),
            sw_batch_size=self._tile_batch_size(1, legacy_batch_size=True),
            sigma_scale=float(self._inference_setting("gaussian_sigma_scale", 0.125)),
        )
