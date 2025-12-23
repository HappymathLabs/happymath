"""
AutoML core module.

Contains utility functions, font support, and evaluation strategies.
"""

from .font import ChineseFontMixin
from .scores import get_scores_strategy
from .utils import (
    DataLike,
    DataLoader,
    ModelStorage,
    StoredModel,
    TargetLike,
    extract_metrics_from_results,
    filter_kwargs_for,
    get_metric_direction,
    is_better_score,
    silence_catboost_artifacts,
)

__all__ = [
    "ChineseFontMixin",
    "DataLike",
    "DataLoader",
    "ModelStorage",
    "StoredModel",
    "TargetLike",
    "extract_metrics_from_results",
    "filter_kwargs_for",
    "get_metric_direction",
    "get_scores_strategy",
    "is_better_score",
    "silence_catboost_artifacts",
]
