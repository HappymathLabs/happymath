"""HappyMath AutoML 对外接口。"""

from .base import AutoMLBase
from .supervised import ClassificationML, RegressionML
from .unsupervised import AnomalyML, ClusteringML
from .time_series import TimeSeriesML

__all__ = [
    "AutoMLBase",
    "ClassificationML",
    "RegressionML",
    "ClusteringML",
    "AnomalyML",
    "TimeSeriesML",
]
