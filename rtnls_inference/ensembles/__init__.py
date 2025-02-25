import json
import os
from pathlib import Path

import torch

from rtnls_inference.ensembles.base import Ensemble
from rtnls_inference.ensembles.ensemble_classification import (  # noqa: F401
    ClassificationEnsemble,
)
from rtnls_inference.ensembles.ensemble_heatmap_regression import (  # noqa: F401
    HeatmapRegressionEnsemble,
)
from rtnls_inference.ensembles.ensemble_keypoints import KeypointsEnsemble  # noqa: F401
from rtnls_inference.ensembles.ensemble_regression import (
    RegressionEnsemble,  # noqa: F401
)
from rtnls_inference.ensembles.ensemble_segmentation import (  # noqa: F401
    SegmentationEnsemble,
)
from rtnls_inference.ensembles.ensemble_segmentation_overlaps import (  # noqa: F401
    SegmentationEnsembleOverlaps,
)
from rtnls_inference.utils import find_release_file, get_all_subclasses_dict

name_to_ensemble = get_all_subclasses_dict(Ensemble)


def get_ensemble_class(config) -> type[Ensemble]:
    from rtnls_models.models import get_model_class

    model_class = get_model_class(config)
    ensemble_class = model_class._ensemble_class
    return ensemble_class


def make_ensemble(release_path: str | Path) -> Ensemble:
    release_file = find_release_file(release_path)

    extra_files = {"config.yaml": ""}  # values will be replaced with data

    ensemble = torch.jit.load(release_file, _extra_files=extra_files).eval()
    config = json.loads(extra_files["config.yaml"])
    ensemble_class = get_ensemble_class(config)
    return ensemble_class(ensemble, config, release_path)


def make_ensemble_name(release_name: str | Path) -> Ensemble:
    return make_ensemble(os.path.join(os.environ["RTNLS_MODEL_RELEASES"], release_name))
