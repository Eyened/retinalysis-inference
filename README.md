# Retinalysis inference

Run model ensembles for ophthalmic image classification, regression, segmentation
and keypoint localisation. The library handles preprocessing, prediction and
postprocessing, and is used by VascX for fundus image analysis.

Models can be loaded from local files or Hugging Face. You can also train your
own models with [retinalysis-models](https://github.com/eyened/retinalysis-models).

## Installation

```bash
pip install retinalysis-inference
```

For GPU use, follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/)
to install a build suitable for your machine.

## Running inference

### Load a model

Load a local release by its path without the file extension and choose a device:

```python
from pathlib import Path

import torch
from rtnls_inference.ensembles import make_ensemble

device = "cuda:0" if torch.cuda.is_available() else "cpu"
ensemble = make_ensemble("/path/to/model").to(device).eval()
```

The [example notebooks](#examples-and-further-reading) also show how to load
pretrained VascX models from Hugging Face.

### Single image

Inference follows four steps:

```python
# 1. Preprocess the image.
item = ensemble.preprocess_item({"image": "fundus.jpg"}, input_space="raw")

# 2. Make a batch of one.
batch = ensemble.make_batch([item])

# 3. Run the model.
predictions = ensemble.predict_batch(batch)

# 4. Postprocess the output.
result = ensemble.postprocess_batch(predictions, batch)[0]
```

Read the prediction from `result["output"]`. You can also pass a loaded uint8 RGB
NumPy image or PyTorch tensor in the `image` field.

Use `input_space="preprocessed"` for images that have already been through fundus
preprocessing. Segmentation masks and keypoints refer to the preprocessed image,
which is available in `result["preprocessed_image"]`.

### Batch inference

Create the input list, build a dataloader, then run inference:

```python
# 1. Create the input list.
data = [{"image": path} for path in sorted(Path("images").glob("*.jpg"))]

# 2. Create the dataloader.
loader = ensemble.make_inference_dataloader(data, input_space="raw", batch_size=2)

# 3. Run inference and save the results.
results = ensemble.predict_dataloader(loader, dest_path="predictions")
```

- **Classification, regression and keypoints** return a DataFrame and save
  `predictions.csv`. Omit `dest_path` to keep the results in memory only.
- **Segmentation** saves masks as it goes and returns a table listing the output
  files, also saved as `manifest.csv`.

To process results one at a time without saving files, use
`ensemble.iter_predictions(loader)`. The notebooks below use this for visualizing
segmentation results.

## Examples and further reading

Each notebook demonstrates single-image inference and a batch visualization:

- [Segmentation](notebooks/inference_segmentation.ipynb): artery/vein and vessel masks.
- [Classification](notebooks/inference_classification.ipynb): quality labels and probabilities.
- [Keypoint localisation](notebooks/inference_keypoints.ipynb): fovea coordinates.
- [ONNX](notebooks/inference_onnx.ipynb): local ONNX segmentation releases.

Find pretrained models on [Hugging Face](https://huggingface.co/Eyened/vascx).
For input formats and additional options, see the method documentation in
[the ensemble base classes](rtnls_inference/ensembles/base.py) and
[the segmentation ensemble](rtnls_inference/ensembles/ensemble_segmentation.py).

For raw logits and migrating older callers, see the short
[inference workflow guide](docs/inference_migration.md).
