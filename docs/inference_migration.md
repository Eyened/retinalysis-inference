# Inference workflows and migration

The public workflow separates input preparation, tensor prediction, per-image
conversion, postprocessing, and optional export. Input records are validated once
when the inference dataset is constructed; image decoding stays lazy.

## Predict and inspect

```python
from rtnls_inference.ensembles import unbatch_predictions

batch = ensemble.make_batch([
    ensemble.preprocess_item({"image": "fundus.jpg"}, input_space="raw")
])
predictions = ensemble.predict_batch(batch, fields=("logits",))
raw = unbatch_predictions(predictions, batch)[0]
results = ensemble.postprocess_batch(predictions, batch)
```

`predict_batch` runs the model once and returns detached tensors on its device.
Without `fields`, it retains only the fields needed for postprocessing. A sequence
adds inspection fields; `fields="all"` keeps every available field. An unavailable
field raises an error. Classification logits are its `aggregate`; segmentation
also exposes averaged `logits`. Member `prediction` and heatmap `heatmaps` are
available when the corresponding model produces them.

`unbatch_predictions` returns NumPy dictionaries at model resolution, with IDs,
canonical images when supplied, and geometry. Prediction arrays have their first
(batch) axis removed. Shared head names and refinement configuration stay intact.
Float16 stays float16; bfloat16 becomes float32 for NumPy.

`postprocess_batch` restores final results to canonical preprocessing coordinates.
It transfers only required fields, even if predictions contain additional
inspection tensors. For AV models, `return_postprocess_intermediates=True` exposes
restored logits and probabilities in the final result; these differ from raw
model-resolution inspection. Raw-logit inspection does not run AV refinement.

## Load and export

```python
records = [{"id": "example", "image": "fundus.jpg"}]
loader = ensemble.make_inference_dataloader(
    records, input_space="raw", batch_size=8, num_workers=0,
)
for result in ensemble.iter_predictions(loader):
    print(result["id"], result["output"].shape)
```

Use `input_space="preprocessed"` for already-preprocessed images. Loading failures
raise by default; pass `ignore_exceptions=True` to warn and skip failed images.
Malformed records still fail during dataset construction.

`predict_dataloader(loader)` collects numeric outputs into a table with an `id`
column and named prediction columns. A destination additionally writes
`predictions.csv`. Segmentation requires a destination and returns/writes
`manifest.csv`; use `iter_predictions` for in-memory segmentation.

Exports reject missing/unsafe IDs, duplicates, and existing output files. Pass
`overwrite=True` for an intentional rerun. Ordinary label masks are PNGs.
Overlapping multilabel masks are boolean HWC `.npy` arrays, preserving every
channel; read these with `numpy.load`. LUNet's encoded AV label mask remains PNG.

## Update older callers

| Removed entry point | Replacement |
| --- | --- |
| `predict_step_full` + `decollate_predict_full` | `predict_batch(..., fields="all")` + `unbatch_predictions(predictions, batch)` |
| Full prediction followed by `postprocess_item` | `postprocess_batch(predict_batch(batch), batch)` |
| `predict`, `predict_dataset` | Public loader with `input_space="raw"`, then `predict_dataloader` |
| `predict_preprocessed` | Public loader with `input_space="preprocessed"`, then `predict_dataloader` |
| `predict_dataframe` | Select input columns, build records, then use the public loader |
| `_make_inference_dataloader` | `make_inference_dataloader(records, ...)`; replace `preprocess` with `input_space` |
| `_predict_batch`, `_predict_dataloader` | `postprocess_batch`, `iter_predictions`, or `predict_dataloader` |

There are no deprecated aliases. Lightning's `predict_step` still returns the
aggregate tensor; general-purpose `utils.decollate_batch` remains available.

DataFrame callers should select only input columns and map missing IDs to `None`:

```python
import pandas as pd

records = [
    {"id": None if pd.isna(row.get("id")) else row.get("id"), "image": row["path"]}
    for _, row in frame.iterrows()
]
```

Keep an existing numeric file format at its save boundary. For example, after
embedding inference:

```python
table = ensemble.predict_dataloader(loader)
embeddings = table.set_index("id").filter(regex="^embedding_")
embeddings.columns = range(len(embeddings.columns))
embeddings.index.name = None
```

Select columns in their returned model order, rather than sorting numbered names
lexicographically. Quality callers similarly select `logit_` columns before
renaming them. Preserve explicit save operations: older numeric wrappers ignored
their destination argument. They also skipped loading failures by default, so
migrated callers that require that behavior must pass `ignore_exceptions=True`.

The training CLI's `predict-devset` command and its export helpers were removed.
Metric evaluation, cross-validation, and checkpoint/release loading remain
available. Load an ensemble from checkpoints and use the public inference
workflow when predictions are needed.
