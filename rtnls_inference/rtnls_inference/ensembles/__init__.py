from rtnls_models.utils.utils import get_all_subclasses_dict

from rtnls_inference.ensembles.base import FundusEnsemble
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

name_to_ensemble = get_all_subclasses_dict(FundusEnsemble)


def get_ensemble_class(config) -> type[FundusEnsemble]:
    from rtnls_models.models import get_model_class

    model_class = get_model_class(config)
    ensemble_class = model_class._ensemble_class
    return ensemble_class
