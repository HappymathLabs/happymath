"""Public interface for HappyMath AutoML."""

import os

# ============================================================================
# 关键：在导入任何 PyCaret 或 CatBoost 相关模块之前设置环境变量
# 这些设置必须在模块级别最早执行
# ============================================================================
os.environ["PYCARET_CUSTOM_LOGGING_PATH"] = os.devnull
os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "CRITICAL"
os.environ["CATBOOST_ALLOW_WRITING_FILES"] = "0"

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
