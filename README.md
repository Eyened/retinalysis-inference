## Retinalysis models inference

This repository implements inference ensembles for retinalysis model releases. It includes the same pre-processing code that was used to train the models. For example, fundus preprocessing will detect bounds, crop the smalles square that contains these bounds, and resize it to a fixed resolution, currently 1024x1024px.

- VascX models are available in [this huggingface repository](https://huggingface.co/Eyened/vascx) but don't need to be downloaded manually. See [this notebook](./notebooks/inference.ipynb).

Models have been tested to run on a single nvidia GPU with at least 10GB VRAM. Using them for distributed inference in multiple GPUs will require some adaptation.

### Inference contracts

Inference has two batch methods because they serve different hardware and consumers:

- `predict_step({"image": tensor}) -> Tensor` is the GPU primitive. It returns the decoded, member-aggregated `N...` tensor without a CUDA synchronization, CPU refinement, resizing, or file I/O.
- `predict_step_full(batch) -> dict` is the inspection primitive. It returns a Pydantic-validated plain dictionary containing per-member NumPy values in `NM...` order, the aggregate in `N...` order, intermediates such as averaged logits, and canonical preprocessing context when the batch came from a package dataloader.

The two methods intentionally expose different arrays: `full["prediction"]` retains the member axis, while `full["aggregate"]` is the NumPy representation of `predict_step`.

Two higher-level methods handle datasets:

- `predict_preprocessed(data, dest_path=None, ...)` accepts images already in canonical fundusprep space.
- `predict_dataframe(df, ..., preprocess=True)` accepts a table and may preprocess raw inputs first.

High-level spatial outputs always end in canonical preprocessed geometry, normally 1024×1024. The package does not convert results back to the original photograph geometry. `predict_step` and raw arrays in `predict_step_full` remain in model prediction geometry; call `postprocess_item` on a decollated full item to run family-specific CPU refinement and resize-only canonical restoration.

`predict` and `predict_dataset` remain undocumented raw-input compatibility entrypoints. `_predict_batch` remains temporarily available as a prediction-geometry adapter and emits `FutureWarning`.

| Family | `prediction` | `aggregate` | Full-output intermediates |
|---|---|---|---|
| Regression/classification | `NMC` | `NC` | — |
| Segmentation | `NMHWC` member logits | `NHWC` softmax | averaged `logits: NHWC` |
| Overlap/LUNet | `NMHWC` member logits | `NHWC` sigmoid | averaged `logits: NHWC` |
| Embedding | `NMD` | `ND` | — |
| Keypoints | `NMK2` | `NK2` | — |
| Heatmap | `NMK2` | `NK2` | optional `heatmaps: NMKHW` |
| Halo artery/vein | `NMHW4` member logits | `NHW4` probabilities | averaged `logits: NHW4`, exact four names and runtime refinement configuration |

### Configuring inference

Runtime settings from `config.inference` can be overridden when constructing an ensemble. Constructor values win over the release defaults. Nested mappings such as `graph_refinement` are merged field by field.

```python
ensemble = make_ensemble(
    release_path,
    tta=False,
    overlap=0.25,
    tile_batch_size=8,
    graph_refinement={"mode": "full", "direction_cost_weight": 0.0},
)
```

- `batch_size` is the image/dataloader batch size.
- `tile_batch_size` is the sliding-window tile micro-batch. Halo models still fall back to embedded `inference.batch_size` when `tile_batch_size` is omitted.
- New ensembles must read these settings through `_inference_setting`, `_inference_mapping`, and `_tile_batch_size`, not from `self.config["inference"]` directly.

For artery/vein models, embedded `inference.graph_refinement` is the default. `refinement_mode` and `refinement_parameters` remain supported aliases and take precedence over `graph_refinement`. Passing `refinement_mode=None` still selects basic per-pixel decoding.

In `full` mode, configurable-radius junction regions expose segment ports. The optimizer scores continuation, bifurcation, overlap, and cut configurations from spline direction, thickness, vessel/crossing logits, and A/V compatibility. A mixed-integer program jointly assigns segment classes and configurations while forcing the artery and vein graphs to be acyclic rooted forests. When `disc_mask` and `bounds_mask` are supplied to `infer_artery_vein`, disc pixels are removed and roots are prioritized as disc contact, fundus-boundary fallback, then internal fallback. The inward fundus boundary width is configurable with `bounds_boundary_width`. Short components below `min_segment_size` are retained as microsegments when they connect at least two junction boundaries. Microsegments use their endpoint chord instead of a spline, cannot become roots, and must connect at every port when active. Individual evidence costs and the disc, bounds, internal, distance, and thickness root costs can be ablated independently by setting their corresponding weights or penalties to zero.

Do not add logit-specific step or bulk aliases. Read logits from `predict_step_full(batch)["logits"]`; a bulk export is a short dataloader loop that restores each array to canonical geometry before writing it.

### Installation

1. Install torch and torchvision that match your cuda environment. For example:
```
pip3 install torch torchvision torchaudio  # pip and CUDA 12
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia # conda and CUDA 12
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # pip and CUDA 11
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia # conda and CUDA 11
```

Pytorch installation instructions are [here](https://pytorch.org/get-started/locally/).

We did not include torch as a dependency of rtnls-inference. These must be installed manually beforehand. 

2. Install rtnls_fundusprep and rtnls-inference:

```
pip install retinalysis-fundusprep
pip install retinalysis-inference
```

3. Done! You can now download and use rtnls-inference ensembles. See [this notebook](./notebooks/inference.ipynb) for an example. The models are automatically downloaded from huggingface.
