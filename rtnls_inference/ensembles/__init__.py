import json
import os
from collections.abc import Sequence
from pathlib import Path

import torch

from rtnls_inference.ensembles.base import Ensemble
from rtnls_inference.ensembles.ensemble_artery_vein import (  # noqa: F401
    ArteryVeinSegmentationEnsemble,
)
from rtnls_inference.ensembles.ensemble_classification import (  # noqa: F401
    ClassificationEnsemble,
)
from rtnls_inference.ensembles.ensemble_embedding import EmbeddingEnsemble  # noqa: F401
from rtnls_inference.ensembles.ensemble_heatmap_regression import (  # noqa: F401
    HeatmapRegressionEnsemble,
)
from rtnls_inference.ensembles.ensemble_keypoints import KeypointsEnsemble  # noqa: F401
from rtnls_inference.ensembles.ensemble_lunet_artery_vein import (  # noqa: F401
    LUNetArteryVeinEnsemble,
)
from rtnls_inference.ensembles.ensemble_regression import (
    RegressionEnsemble,  # noqa: F401
)
from rtnls_inference.ensembles.ensemble_segmentation import (  # noqa: F401
    SegmentationEnsemble,
)
from rtnls_inference.ensembles.ensemble_segmentation_overlaps import (  # noqa: F401
    SegmentationEnsembleOverlaps,
)
from rtnls_inference.ensembles.onnx_backend import (
    OnnxEnsembleBackend,
    load_config_from_onnx,
)
from rtnls_inference.release_config import load_stored_config, update_stored_config
from rtnls_inference.utils import find_release_file, get_all_subclasses_dict

name_to_ensemble = get_all_subclasses_dict(Ensemble)


def get_ensemble_class(config) -> type[Ensemble]:
    """Resolve the inference wrapper class from embedded release config."""
    ensemble_name = config.get("inference", {}).get("ensemble_class")
    if ensemble_name:
        try:
            return name_to_ensemble[ensemble_name]
        except KeyError as exc:
            known = ", ".join(sorted(name_to_ensemble))
            raise ValueError(
                f"Unknown ensemble class {ensemble_name!r}. Known classes: {known}"
            ) from exc

    try:
        from rtnls_models.models import get_model_class
    except ImportError as exc:
        raise ImportError(
            "Release config has no inference.ensemble_class and rtnls_models is "
            "not installed. Install rtnls-models or re-export the release."
        ) from exc

    model_class = get_model_class(config)
    ensemble_class = model_class._ensemble_class
    if ensemble_class is None:
        raise ValueError(f"Model class {model_class.__name__} has no _ensemble_class")
    return ensemble_class


def make_ensemble(release_path: str | Path, prefer: str | None = None) -> Ensemble:
    release_file = find_release_file(release_path, prefer=prefer)

    if release_file.suffix == ".onnx":
        backend = OnnxEnsembleBackend(release_file).eval()
        config = backend.config
        ensemble = backend
    else:
        extra_files = {"config.yaml": ""}
        ensemble = torch.jit.load(release_file, _extra_files=extra_files).eval()
        config = json.loads(extra_files["config.yaml"])

    ensemble_class = get_ensemble_class(config)
    return ensemble_class(ensemble, config, release_path)


def make_ensemble_name(
    release_name: str | Path,
    prefer: str | None = None,
) -> Ensemble:
    return make_ensemble(
        os.path.join(os.environ["RTNLS_MODEL_RELEASES"], release_name),
        prefer=prefer,
    )


def make_ensemble_from_checkpoints(
    checkpoint_name: str | None = None,
    checkpoints: str | Path | Sequence[str | Path] | None = None,
    map_location: str | torch.device = "cpu",
) -> Ensemble:
    """Load an inference ensemble directly from training checkpoints."""
    try:
        from rtnls_models.wrapper import EnsembleWrapper
    except ImportError as exc:
        raise ImportError(
            "Loading checkpoints requires the optional runtime dependency "
            "`rtnls_models`. Install rtnls-models or use a TorchScript release."
        ) from exc

    if checkpoints is None:
        if checkpoint_name is None:
            raise ValueError("checkpoint_name is required when checkpoints is not set")
        wrapper = EnsembleWrapper.from_checkpoint(
            checkpoint_name, map_location=map_location
        )
        source = checkpoint_name
    else:
        checkpoint_paths = (
            [checkpoints] if isinstance(checkpoints, (str, Path)) else list(checkpoints)
        )
        wrapper = EnsembleWrapper.from_checkpoint_files(
            checkpoint_paths,
            map_location=map_location,
            checkpoint_name=checkpoint_name,
        )
        source = checkpoint_paths[0] if len(checkpoint_paths) == 1 else checkpoint_paths

    wrapper = wrapper.eval()
    ensemble_class = get_ensemble_class(wrapper.config)
    return ensemble_class(wrapper, wrapper.config, source)
