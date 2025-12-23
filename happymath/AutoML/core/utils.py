"""
AutoML utility module.

Contains data loading, metrics utilities, CatBoost patches, and model storage.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# =============================================================================
# Type Definitions
# =============================================================================

DataLike = Union[str, pd.DataFrame, np.ndarray, Tuple[np.ndarray, np.ndarray]]
TargetLike = Union[str, int, None]


# =============================================================================
# CatBoost Patch
# =============================================================================

_catboost_patched = False


def silence_catboost_artifacts() -> None:
    """Disable CatBoost file writing via monkey patching."""
    global _catboost_patched
    if _catboost_patched:
        return

    try:
        import catboost  # type: ignore
    except ImportError:
        return

    def _patch_init(cls: Any) -> None:
        original_init = cls.__init__
        sig = inspect.signature(original_init)
        param_names = set(sig.parameters.keys())

        def wrapped(self, *args: Any, **kwargs: Any):
            if "allow_writing_files" in param_names:
                kwargs.setdefault("allow_writing_files", False)
            return original_init(self, *args, **kwargs)

        wrapped.__signature__ = sig  # type: ignore
        cls.__init__ = wrapped  # type: ignore

    for name in ("CatBoostRegressor", "CatBoostClassifier"):
        model_cls = getattr(catboost, name, None)
        if model_cls is not None:
            _patch_init(model_cls)

    _catboost_patched = True


# =============================================================================
# Data Loader
# =============================================================================


class DataLoader:
    """Data loading and normalization utility class."""

    @staticmethod
    def is_sklearn_bunch(data: Any) -> bool:
        """Check if data is a sklearn Bunch object (from load_*, fetch_* datasets)."""
        return (
            hasattr(data, 'data') and
            hasattr(data, 'target') and
            hasattr(data, 'feature_names')
        )

    @staticmethod
    def handle_sklearn_bunch(
        data: Any,
        target: TargetLike
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Handle sklearn Bunch objects (load_*, fetch_* datasets).

        Args:
            data: sklearn Bunch object with .data, .target, .feature_names attributes
            target: None, string column name, integer index, or tuple index

        Returns:
            Tuple of (DataFrame, target_column_name)
        """
        features_data = data.data
        target_data = data.target

        if hasattr(features_data, 'values'):
            features_data = features_data.values
        if hasattr(target_data, 'values'):
            target_data = target_data.values

        if hasattr(data, 'feature_names') and data.feature_names:
            feature_columns = list(data.feature_names)
        else:
            feature_columns = [f"feature_{idx}" for idx in range(features_data.shape[1])]

        df = pd.DataFrame(features_data, columns=feature_columns)

        if target is None:
            target_column_name = "target"
            df[target_column_name] = target_data
            return df, target_column_name
        elif isinstance(target, str):
            target_column_name = target
            df[target_column_name] = target_data
            return df, target_column_name
        elif isinstance(target, int):
            if target == -1:
                target_column_name = "target"
                df[target_column_name] = target_data
                return df, target_column_name
            elif 0 <= target < len(feature_columns):
                target_column_name = feature_columns[target]
                df.rename(columns={target_column_name: "target"}, inplace=True)
                return df, "target"
            else:
                raise ValueError(f"Target column index {target} out of range for {len(feature_columns)} features")
        else:
            raise TypeError("Target must be None, string column name, or integer index")

    @staticmethod
    def load(
        data: DataLike,
        target: TargetLike,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Normalize various input data formats into a DataFrame and handle target column.
        Automatically handles sklearn datasets (Bunch objects, tuples, etc.).
        """
        if DataLoader.is_sklearn_bunch(data):
            return DataLoader.handle_sklearn_bunch(data, target)

        if isinstance(data, str):
            if data.lower().endswith(".csv"):
                df = pd.read_csv(data)
            elif data.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(data)
            else:
                raise ValueError(f"Unsupported file format: {data}")
        elif isinstance(data, pd.Series):
            df = data.to_frame()
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, (np.ndarray, tuple)):
            return DataLoader.convert_array_to_frame(data, target)
        else:
            raise TypeError("data must be a file path, DataFrame, NumPy array, or tuple of (features, target)")

        target_name: Optional[str]
        if target is None:
            target_name = None
        elif isinstance(target, str):
            if target not in df.columns:
                raise ValueError(f"target column '{target}' not found in data")
            target_name = target
        elif isinstance(target, int):
            try:
                target_name = df.columns[target]
            except IndexError as exc:
                raise ValueError(f"target column index {target} out of range") from exc
        else:
            raise TypeError("target must be a column name, index, or None")

        return df, target_name

    @staticmethod
    def convert_array_to_frame(
        data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        target: TargetLike,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Convert NumPy arrays to a DataFrame and handle target column.

        Supports multiple input modes:
        1. Single array with target index
        2. Single array with separate target array/Series
        3. Tuple of (features, target) arrays
        4. Single array with no target

        Note: Feature arrays must contain only numeric data. For mixed-type data,
        use pandas DataFrame instead.
        """
        if isinstance(data, np.ndarray) and not isinstance(data, tuple):
            if isinstance(target, (np.ndarray, pd.Series)):
                target_array = np.asarray(target) if isinstance(target, pd.Series) else target
                data = (data, target_array)
                target = None

        if isinstance(data, tuple):
            if len(data) != 2:
                raise ValueError("When data is a tuple, it must contain exactly 2 arrays: (features, target)")

            features_array, target_array = data

            if not isinstance(features_array, np.ndarray) or not isinstance(target_array, np.ndarray):
                raise TypeError("Both elements of the tuple must be NumPy arrays")

            if features_array.ndim != 2:
                raise ValueError("Features array must be 2-dimensional")

            if target_array.ndim != 1:
                raise ValueError("Target array must be 1-dimensional")

            if len(features_array) != len(target_array):
                raise ValueError("Features and target arrays must have the same length")

            if not np.issubdtype(features_array.dtype, np.number):
                raise TypeError(
                    "Features array must contain only numeric data (int, float, etc.). "
                    "For mixed-type data containing strings or categorical values, "
                    "please use pandas DataFrame instead."
                )

            feature_columns = [f"feature_{idx}" for idx in range(features_array.shape[1])]
            features_df = pd.DataFrame(features_array, columns=feature_columns)

            if target is None:
                target_column_name = "target"
            elif isinstance(target, str):
                target_column_name = target
            elif isinstance(target, int):
                if target == -1:
                    target_column_name = "target"
                elif 0 <= target < len(feature_columns):
                    target_column_name = feature_columns[target]
                else:
                    raise ValueError(f"Target index {target} out of range for {len(feature_columns)} features")
            else:
                raise TypeError("Target must be None, string column name, or integer index")

            features_df[target_column_name] = target_array
            return features_df, target_column_name

        elif isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError("Only 2-D arrays are supported as data input")

            if not np.issubdtype(data.dtype, np.number):
                raise TypeError(
                    "NumPy array must contain only numeric data (int, float, etc.). "
                    "For mixed-type data containing strings or categorical values, "
                    "please use pandas DataFrame instead."
                )

            columns = [f"feature_{idx}" for idx in range(data.shape[1])]
            df = pd.DataFrame(data, columns=columns)

            if target is None:
                return df, None

            if not isinstance(target, int):
                raise TypeError(
                    "In single array mode, target must be None or an integer index. "
                    "To pass a separate target array, use data=X_array, target=y_array."
                )

            try:
                target_column = columns[target]
            except IndexError as exc:
                raise ValueError(f"Target column index {target} out of range") from exc

            df.rename(columns={target_column: "target"}, inplace=True)
            return df, "target"

        else:
            raise TypeError("data must be a NumPy array or tuple of (features, target) arrays")

    @staticmethod
    def validate(data: pd.DataFrame, target: Optional[str]) -> None:
        """Basic data validation to ensure target exists and no duplicate columns."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("internal data must be a pandas.DataFrame")

        if data.columns.duplicated().any():
            raise ValueError("duplicate column names found; please resolve them first")

        if target is not None and target not in data.columns:
            raise ValueError("specified target column not found in data")


# =============================================================================
# Metrics Utilities
# =============================================================================

LOWER_BETTER_METRICS = {
    "MAE",
    "MSE",
    "RMSE",
    "RMSLE",
    "MAPE",
    "MedAE",
    "MASE",
    "RMSSE",
    "SMAPE",
    "Log Loss",
    "FNR",
    "FPR",
}

HIGHER_BETTER_METRICS = {
    "Accuracy",
    "AUC",
    "Recall",
    "Prec.",
    "F1",
    "Kappa",
    "MCC",
    "R2",
    "TPR",
    "TNR",
    "PPV",
    "NPV",
    "Silhouette",
}


def get_metric_direction(metric: str) -> str:
    """Determine metric optimization direction from common names."""
    if metric in LOWER_BETTER_METRICS:
        return "lower_better"
    if metric in HIGHER_BETTER_METRICS:
        return "higher_better"

    print(f"Warning: unknown metric '{metric}', defaulting to higher-is-better")
    return "higher_better"


def is_better_score(
    new_score: float,
    current_best: Optional[float],
    metric: str
) -> bool:
    """Determine whether a new score is better based on metric direction."""
    if current_best is None:
        return True

    direction = get_metric_direction(metric)
    if direction == "higher_better":
        return new_score > current_best
    return new_score < current_best


def extract_metrics_from_results(
    results: Optional[Any],
    model_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract metrics from PyCaret's pull() output."""
    if results is None:
        return {}

    if isinstance(results, dict):
        return dict(results)

    if isinstance(results, pd.Series):
        return results.to_dict()

    if isinstance(results, pd.DataFrame):
        df = results.copy()

        if "Model" in df.columns and model_label:
            matched = df[df["Model"] == model_label]
            if not matched.empty:
                return matched.iloc[0].to_dict()

        index = df.index
        if isinstance(index, pd.Index):
            for key in ("Mean", "Holdout", "Score"):
                if key in index:
                    row = df.loc[key]
                    return row.to_dict()

        if df.shape[0] > 0:
            return df.iloc[-1].to_dict()

    return {}


def filter_kwargs_for(func: Any, base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter unsupported kwargs by function signature."""
    params = inspect.signature(func).parameters
    return {key: value for key, value in base_kwargs.items() if key in params}


def filter_metrics_columns(
    df: pd.DataFrame,
    metrics: Union[str, List[str]]
) -> pd.DataFrame:
    """
    Filter DataFrame columns based on user-specified metrics parameter.

    For columns containing suffixes (e.g. ``Accuracy_train`` / ``Accuracy_test`` in kfold),
    matches with metrics case-insensitively using the base name without suffix.
    """
    if metrics is None or metrics == "all":
        return df

    if isinstance(metrics, str):
        requested = {metrics.lower()}
    else:
        requested = {str(m).lower() for m in metrics}

    def base_name(col: str) -> str:
        for suffix in ("_train", "_test", "_train_mean", "_test_mean"):
            if col.endswith(suffix):
                return col[: -len(suffix)]
        return col

    selected_cols: List[str] = []
    available_bases = {base_name(col).lower() for col in df.columns}

    for col in df.columns:
        if base_name(col).lower() in requested:
            selected_cols.append(col)

    missing = requested - available_bases
    if missing:
        print(f"Warning: The following metrics are not found in current results and will be ignored: {sorted(missing)}")

    if not selected_cols:
        return df
    return df[selected_cols]


# =============================================================================
# Model Storage
# =============================================================================


@dataclass
class StoredModel:
    """Simple container to store model-related information."""

    model: Any
    metrics: Dict[str, Any]
    name: str
    extra: Dict[str, Any]
    timestamp: pd.Timestamp


class ModelStorage:
    """Model storage manager."""

    def __init__(self):
        self._models: Dict[str, StoredModel] = {}

    def store(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store model with its metrics."""
        info = StoredModel(
            model=model,
            metrics=metrics,
            name=model_name,
            extra=extra if extra is not None else {},
            timestamp=pd.Timestamp.now(),
        )
        self._models[model_name] = info

    def get_best(
        self,
        primary_metric: str,
        is_better_fn
    ) -> Tuple[Any, Dict[str, Any]]:
        """Get the best model based on primary_metric."""
        if not self._models:
            raise ValueError("No comparable models; please create or compare models first")

        best_name = None
        best_info: Optional[StoredModel] = None
        best_score: Optional[float] = None

        for name, info in self._models.items():
            score = info.metrics.get(primary_metric)
            if score is None:
                continue
            if is_better_fn(score, best_score):
                best_score = score
                best_name = name
                best_info = info

        if best_info is None:
            fallback = list(self._models.values())[-1]
            print(
                f"Warning: No model contains the primary metric '{primary_metric}', will return the most recent model '{fallback.name}'"
            )
            return fallback.model, fallback.metrics

        return best_info.model, best_info.metrics

    def get_all_names(self) -> List[str]:
        """Get all stored model names."""
        return list(self._models.keys())

    def get(self, name: str) -> Optional[StoredModel]:
        """Get model by name."""
        return self._models.get(name)

    @property
    def models(self) -> Dict[str, StoredModel]:
        """Access internal model dictionary (backward compatible)."""
        return self._models
