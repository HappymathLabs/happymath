"""
AutoML scores evaluation strategies.

Contains strategy classes for evaluating models across different task types:
- Supervised learning (classification/regression)
- Time series forecasting
- Clustering
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, train_test_split

from .utils import extract_metrics_from_results, filter_metrics_columns

if TYPE_CHECKING:
    from ..base import AutoMLBase


# =============================================================================
# Base Strategy
# =============================================================================


class ScoresStrategy(ABC):
    """Base class for evaluation strategies."""

    def __init__(self, automl_instance: "AutoMLBase"):
        """
        Args:
            automl_instance: AutoMLBase instance reference for accessing necessary attributes
        """
        self._automl = automl_instance

    @property
    def experiment(self):
        return self._automl.experiment

    @property
    def data(self) -> pd.DataFrame:
        return self._automl.data

    @property
    def target(self) -> Optional[str]:
        return self._automl.target

    @property
    def current_model(self):
        return self._automl.current_model

    @property
    def test_data(self) -> Optional[pd.DataFrame]:
        return self._automl.test_data

    @property
    def setup_kwargs(self) -> Dict[str, Any]:
        return self._automl.setup_kwargs

    @property
    def verbose(self) -> bool:
        return self._automl.verbose

    @property
    def models(self) -> Dict[str, Any]:
        return self._automl.models

    @abstractmethod
    def evaluate(
        self,
        mode: str,
        metrics: Union[str, List[str]],
        external_test_data: Optional[pd.DataFrame],
        train_size: float,
        fold: int,
    ) -> pd.DataFrame:
        """Execute evaluation and return results DataFrame."""
        pass

    def _get_random_state(self) -> Optional[int]:
        """Get random seed."""
        from_kwargs = self.setup_kwargs.get("session_id", None)
        if from_kwargs is not None:
            return from_kwargs
        return getattr(self._automl, "seed", None)

    def _filter_metrics_columns(
        self,
        df: pd.DataFrame,
        metrics: Union[str, List[str]]
    ) -> pd.DataFrame:
        """Filter metrics columns."""
        return filter_metrics_columns(df, metrics)


# =============================================================================
# Supervised Learning Strategy
# =============================================================================


class SupervisedScoresStrategy(ScoresStrategy):
    """Evaluation strategy for supervised learning (classification/regression)."""

    ALLOWED_MODES = {"auto", "holdout", "kfold", "leaveout", "custom", "train-only"}

    def evaluate(
        self,
        mode: str,
        metrics: Union[str, List[str]],
        external_test_data: Optional[pd.DataFrame],
        train_size: float,
        fold: int,
    ) -> pd.DataFrame:
        """Execute supervised learning evaluation."""
        if mode not in self.ALLOWED_MODES:
            raise ValueError(f"Unknown evaluation mode '{mode}', available options: {sorted(self.ALLOWED_MODES)}")

        has_any_test = external_test_data is not None or self.test_data is not None

        # Auto mode selects based on test set availability and sample size
        if mode == "auto":
            if has_any_test:
                resolved_mode = "custom"
            else:
                n_samples = len(self.data)
                if n_samples < 100:
                    resolved_mode = "leaveout"
                elif n_samples <= 10000:
                    resolved_mode = "kfold"
                else:
                    resolved_mode = "holdout"
        else:
            resolved_mode = mode

        if resolved_mode == "holdout":
            result = self._eval_holdout(metrics, external_test_data, train_size)
        elif resolved_mode == "custom":
            result = self._eval_custom(metrics, external_test_data)
        elif resolved_mode == "train-only":
            result = self._eval_train_only(metrics, external_test_data)
        elif resolved_mode == "kfold":
            result = self._eval_kfold(metrics, fold)
        elif resolved_mode == "leaveout":
            result = self._eval_leaveout(metrics)
        else:
            raise RuntimeError(f"Internal error: unhandled supervised learning evaluation mode '{resolved_mode}'")

        return result

    def _ensure_target(self) -> str:
        """Ensure current task is supervised learning and target column exists."""
        if self.target is None:
            raise ValueError("Current task has no target column configured, cannot evaluate as supervised learning")
        if self.target not in self.data.columns:
            raise ValueError(f"Target column '{self.target}' does not exist in internal data")
        return self.target

    def _eval_split(self, data_split: pd.DataFrame) -> Dict[str, float]:
        """Call PyCaret's predict_model on given data subset and extract numeric metrics."""
        if self.current_model is None:
            raise ValueError("No evaluable model currently available, please create or train a model first")

        self.experiment.predict_model(
            estimator=self.current_model,
            data=data_split,
            verbose=self.verbose,
        )
        results_df = self.experiment.pull()
        metrics_dict = extract_metrics_from_results(results_df, model_label=None)

        numeric_metrics: Dict[str, float] = {}
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float, np.floating)):
                numeric_metrics[key] = float(value)
        return numeric_metrics

    def _eval_holdout(
        self,
        metrics: Union[str, List[str]],
        external_test_data: Optional[pd.DataFrame],
        train_size: float,
    ) -> pd.DataFrame:
        """Supervised learning holdout mode: split train/test sets once according to train_size."""
        target_col = self._ensure_target()
        df = self.data

        random_state = self._get_random_state()

        # Classification tasks prioritize stratified sampling
        usecase = getattr(self.experiment, "_ml_usecase", None)
        usecase_str = str(usecase) if usecase is not None else ""
        stratify = None
        if "CLASSIFICATION" in usecase_str:
            y = df[target_col]
            stratify = y
        try:
            train_df, test_df = train_test_split(
                df,
                train_size=train_size,
                random_state=random_state,
                stratify=stratify,
            )
        except ValueError:
            # Fall back to regular splitting when stratification fails
            train_df, test_df = train_test_split(
                df,
                train_size=train_size,
                random_state=random_state,
            )

        # Override with external test data if provided
        if external_test_data is not None:
            if target_col not in external_test_data.columns:
                raise ValueError(
                    f"Custom test set missing target column '{target_col}', cannot calculate supervised task evaluation metrics"
                )
            test_df = external_test_data

        train_metrics = self._eval_split(train_df)
        test_metrics = self._eval_split(test_df)

        result_df = pd.DataFrame(
            [train_metrics, test_metrics],
            index=["train", "test"],
        )
        return self._filter_metrics_columns(result_df, metrics)

    def _eval_custom(
        self,
        metrics: Union[str, List[str]],
        external_test_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Supervised learning custom mode: training set is initialization data, test set from parameters/attributes."""
        target_col = self._ensure_target()
        train_df = self.data

        if external_test_data is not None:
            test_df = external_test_data
        elif self.test_data is not None:
            test_df = self.test_data
        else:
            raise ValueError(
                "Custom mode must provide test set: please pass in scores(test_data=...), or provide test_data when initializing AutoML."
            )

        if target_col not in test_df.columns:
            raise ValueError(
                f"Custom test set missing target column '{target_col}', cannot calculate supervised task evaluation metrics"
            )

        train_metrics = self._eval_split(train_df)
        test_metrics = self._eval_split(test_df)

        result_df = pd.DataFrame(
            [train_metrics, test_metrics],
            index=["train", "test"],
        )
        return self._filter_metrics_columns(result_df, metrics)

    def _eval_train_only(
        self,
        metrics: Union[str, List[str]],
        external_test_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Supervised learning train-only mode: always evaluate training set, optionally test set."""
        target_col = self._ensure_target()
        train_df = self.data

        rows: List[Dict[str, float]] = []
        indices: List[str] = []

        train_metrics = self._eval_split(train_df)
        rows.append(train_metrics)
        indices.append("train")

        # Check if test set is available
        test_df: Optional[pd.DataFrame] = None
        if external_test_data is not None:
            test_df = external_test_data
        elif self.test_data is not None:
            test_df = self.test_data

        if test_df is not None:
            if target_col not in test_df.columns:
                raise ValueError(
                    f"Custom test set missing target column '{target_col}', cannot calculate supervised task evaluation metrics"
                )
            test_metrics = self._eval_split(test_df)
            rows.append(test_metrics)
            indices.append("test")

        result_df = pd.DataFrame(rows, index=indices)
        return self._filter_metrics_columns(result_df, metrics)

    def _eval_kfold(
        self,
        metrics: Union[str, List[str]],
        fold: int,
    ) -> pd.DataFrame:
        """Supervised learning kfold mode: evaluate train/test subsets for each fold and provide mean."""
        target_col = self._ensure_target()
        df = self.data

        usecase = getattr(self.experiment, "_ml_usecase", None)
        usecase_str = str(usecase) if usecase is not None else ""

        random_state = self._get_random_state()
        if "CLASSIFICATION" in usecase_str:
            y = df[target_col]
            splitter = StratifiedKFold(
                n_splits=fold,
                shuffle=True,
                random_state=random_state,
            )
            split_iter = splitter.split(df, y)
        else:
            splitter = KFold(
                n_splits=fold,
                shuffle=True,
                random_state=random_state,
            )
            split_iter = splitter.split(df)

        rows: List[Dict[str, float]] = []
        indices: List[str] = []

        sum_accumulator: Dict[str, float] = {}
        n_folds_actual = 0

        for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            train_metrics = self._eval_split(train_df)
            test_metrics = self._eval_split(test_df)

            row: Dict[str, float] = {}
            all_keys = set(train_metrics.keys()) | set(test_metrics.keys())
            for key in all_keys:
                if key in train_metrics:
                    row[f"{key}_train"] = train_metrics[key]
                if key in test_metrics:
                    row[f"{key}_test"] = test_metrics[key]

            rows.append(row)
            indices.append(f"fold_{fold_idx + 1}")
            n_folds_actual += 1

            for k, v in row.items():
                sum_accumulator[k] = sum_accumulator.get(k, 0.0) + v

        if n_folds_actual == 0:
            raise ValueError("Kfold evaluation produced no folds, please check data volume and fold configuration")

        mean_row = {k: v / n_folds_actual for k, v in sum_accumulator.items()}
        rows.append(mean_row)
        indices.append("mean")

        result_df = pd.DataFrame(rows, index=indices)
        return self._filter_metrics_columns(result_df, metrics)

    def _eval_leaveout(
        self,
        metrics: Union[str, List[str]],
    ) -> pd.DataFrame:
        """Supervised learning leaveout mode: leave-one-out cross-validation."""
        target_col = self._ensure_target()
        df = self.data
        n_samples = len(df)
        if n_samples <= 1:
            raise ValueError("Leave-one-out cross-validation requires at least 2 samples")

        loo = LeaveOneOut()

        train_sum: Dict[str, float] = {}
        test_sum: Dict[str, float] = {}
        n_rounds = 0

        for train_idx, test_idx in loo.split(df):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            train_metrics = self._eval_split(train_df)
            test_metrics = self._eval_split(test_df)

            for k, v in train_metrics.items():
                train_sum[k] = train_sum.get(k, 0.0) + v
            for k, v in test_metrics.items():
                test_sum[k] = test_sum.get(k, 0.0) + v
            n_rounds += 1

        if n_rounds == 0:
            raise ValueError("Leave-one-out cross-validation produced no rounds, please check data configuration")

        all_keys = set(train_sum.keys()) | set(test_sum.keys())
        train_mean = {k: train_sum.get(k, 0.0) / n_rounds for k in all_keys}
        test_mean = {k: test_sum.get(k, 0.0) / n_rounds for k in all_keys}

        result_df = pd.DataFrame(
            [train_mean, test_mean],
            index=["train_mean", "test_mean"],
        )
        return self._filter_metrics_columns(result_df, metrics)


# =============================================================================
# Time Series Strategy
# =============================================================================


class TimeSeriesScoresStrategy(ScoresStrategy):
    """Evaluation strategy for time series tasks."""

    ALLOWED_MODES = {"auto", "holdout", "kfold", "custom", "train-only"}

    def evaluate(
        self,
        mode: str,
        metrics: Union[str, List[str]],
        external_test_data: Optional[pd.DataFrame],
        train_size: float,
        fold: int,
    ) -> pd.DataFrame:
        """Execute time series evaluation."""
        if mode == "leaveout":
            raise ValueError("Time series tasks currently do not support leaveout mode")

        if mode not in self.ALLOWED_MODES:
            raise ValueError(f"Time series task does not support evaluation mode '{mode}', available options: {sorted(self.ALLOWED_MODES)}")

        has_any_test = external_test_data is not None or self.test_data is not None

        if mode == "auto":
            if has_any_test:
                resolved_mode = "custom"
            else:
                n_samples = len(self.data)
                if n_samples <= 10000:
                    resolved_mode = "kfold"
                else:
                    resolved_mode = "holdout"
        else:
            resolved_mode = mode

        if resolved_mode == "kfold":
            result = self._eval_kfold(metrics)
        elif resolved_mode == "holdout":
            result = self._eval_holdout(metrics, external_test_data)
        elif resolved_mode == "custom":
            result = self._eval_custom(metrics, external_test_data)
        elif resolved_mode == "train-only":
            result = self._eval_train_only(metrics, external_test_data)
        else:
            raise RuntimeError(f"Internal error: unhandled time series evaluation mode '{resolved_mode}'")

        return result

    def _get_cv_results(self) -> Optional[pd.DataFrame]:
        """Try to retrieve cross-validation results table for current time series model."""
        usecase = getattr(self.experiment, "_ml_usecase", None)
        if usecase is None or "TIME_SERIES" not in str(usecase):
            return None

        # Priority: find record matching current model object
        for info in self.models.values():
            if info.model is self.current_model and isinstance(info.extra, dict):
                cv_df = info.extra.get("ts_cv_results")
                if isinstance(cv_df, pd.DataFrame):
                    return cv_df

        # Fallback: search backwards chronologically
        for info in reversed(list(self.models.values())):
            if isinstance(info.extra, dict):
                cv_df = info.extra.get("ts_cv_results")
                if isinstance(cv_df, pd.DataFrame):
                    return cv_df

        return None

    def _get_train_metrics(self) -> Dict[str, float]:
        """Get metrics used to represent training performance in time series tasks."""
        cv_df = self._get_cv_results()
        if cv_df is not None:
            tmp = cv_df.copy()
            metrics_series = None
            if any(str(idx).lower() == "mean" for idx in tmp.index):
                metrics_series = tmp.loc[[idx for idx in tmp.index if str(idx).lower() == "mean"][0]]
            else:
                if any(str(idx).lower() == "sd" for idx in tmp.index):
                    tmp = tmp.drop(index=[idx for idx in tmp.index if str(idx).lower() == "sd"][0])
                metrics_series = tmp.mean(axis=0)

            drop_cols = [col for col in ["cutoff"] if col in metrics_series.index]
            metrics_series = metrics_series.drop(labels=drop_cols)

            metrics_dict: Dict[str, float] = {}
            for key, value in metrics_series.items():
                if isinstance(value, (int, float, np.floating)):
                    metrics_dict[key] = float(value)
            if metrics_dict:
                return metrics_dict

        # Fallback: find current model's metrics in stored models
        for info in self.models.values():
            if info.model is self.current_model:
                numeric_metrics: Dict[str, float] = {}
                for key, value in info.metrics.items():
                    if isinstance(value, (int, float, np.floating)):
                        numeric_metrics[key] = float(value)
                if numeric_metrics:
                    return numeric_metrics

        raise ValueError("Current time series model lacks training phase metric information, cannot construct train portion of scores")

    def _get_internal_test_df(self) -> Optional[pd.DataFrame]:
        """Try to get internal y_test from PyCaret configuration."""
        try:
            y_test = self.experiment.get_config("y_test")
        except Exception:
            return None

        if y_test is None:
            return None

        target_col = self.target
        if target_col is None:
            target_col = "y"
        return pd.DataFrame({target_col: y_test})

    def _eval_on_test(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Use current model to predict on given test set and manually calculate metrics."""
        if self.current_model is None:
            raise ValueError("No evaluable time series model currently available")

        target_col = self.target
        if target_col is not None and target_col in test_df.columns:
            y_true = test_df[target_col]
        else:
            if target_col is None and test_df.shape[1] == 1:
                y_true = test_df.iloc[:, 0]
            else:
                raise ValueError(
                    "Time series test set lacks available target column, cannot calculate evaluation metrics"
                )

        n_periods = len(y_true)
        if n_periods <= 0:
            raise ValueError("Time series test set is empty, cannot calculate evaluation metrics")

        try:
            from sktime.forecasting.base import ForecastingHorizon
        except ImportError as exc:
            raise ImportError(
                "Time series scores requires sktime support, please ensure PyCaret with time series dependencies is installed."
            ) from exc

        fh = ForecastingHorizon(np.arange(1, n_periods + 1), is_relative=True)
        model = self.current_model

        y_pred = model.predict(fh=fh)

        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        m = min(len(y_true_arr), len(y_pred_arr))
        if m == 0:
            raise ValueError("Time series test set and prediction results cannot be aligned, evaluation failed")
        y_true_arr = y_true_arr[:m]
        y_pred_arr = y_pred_arr[:m]

        metrics_containers = getattr(self.experiment, "_all_metrics", {})
        scores: Dict[str, float] = {}
        for key, container in metrics_containers.items():
            display_name = getattr(container, "display_name", key)
            score_func = getattr(container, "score_func", None)
            if score_func is None:
                continue
            try:
                value = score_func(y_true_arr, y_pred_arr)
            except Exception:
                continue
            if isinstance(value, (int, float, np.floating)):
                scores[display_name] = float(value)

        return scores

    def _eval_kfold(self, metrics: Union[str, List[str]]) -> pd.DataFrame:
        """Time series kfold mode: construct folds and average results based on ts_cv_results."""
        cv_df = self._get_cv_results()
        if cv_df is None:
            raise ValueError(
                "Backtesting results for current time series model not found, cannot calculate scores in kfold mode."
            )

        df = cv_df.copy()

        if "cutoff" in df.columns:
            df = df.drop(columns=["cutoff"])

        fold_mask = ~df.index.astype(str).isin(["Mean", "SD"])
        fold_rows = df[fold_mask]

        if fold_rows.empty:
            raise ValueError("No valid fold information found in time series backtesting results")

        if any(str(idx).lower() == "mean" for idx in df.index):
            mean_row = df.loc[[idx for idx in df.index if str(idx).lower() == "mean"][0]]
        else:
            tmp = fold_rows
            if any(str(idx).lower() == "sd" for idx in tmp.index):
                tmp = tmp.drop(index=[idx for idx in tmp.index if str(idx).lower() == "sd"][0])
            mean_row = tmp.mean(axis=0)

        rows: List[Dict[str, float]] = []
        indices: List[str] = []

        for i, (_, row) in enumerate(fold_rows.iterrows(), start=1):
            row_dict: Dict[str, float] = {}
            for key, value in row.items():
                if isinstance(value, (int, float, np.floating)):
                    row_dict[key] = float(value)
            rows.append(row_dict)
            indices.append(f"fold_{i}")

        mean_dict: Dict[str, float] = {}
        for key, value in mean_row.items():
            if isinstance(value, (int, float, np.floating)):
                mean_dict[key] = float(value)
        rows.append(mean_dict)
        indices.append("mean")

        result_df = pd.DataFrame(rows, index=indices)
        return self._filter_metrics_columns(result_df, metrics)

    def _eval_holdout(
        self,
        metrics: Union[str, List[str]],
        external_test_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Time series holdout mode."""
        train_metrics = self._get_train_metrics()

        if external_test_data is not None:
            test_df = external_test_data
        elif self.test_data is not None:
            test_df = self.test_data
        else:
            internal_test_df = self._get_internal_test_df()
            if internal_test_df is None:
                raise ValueError(
                    "No available test set found in time series holdout mode: "
                    "Neither scores(test_data=...) provided nor y_test configured in experiment."
                )
            test_df = internal_test_df

        test_metrics = self._eval_on_test(test_df)

        result_df = pd.DataFrame(
            [train_metrics, test_metrics],
            index=["train", "test"],
        )
        return self._filter_metrics_columns(result_df, metrics)

    def _eval_custom(
        self,
        metrics: Union[str, List[str]],
        external_test_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Time series custom mode."""
        train_metrics = self._get_train_metrics()

        if external_test_data is not None:
            test_df = external_test_data
        elif self.test_data is not None:
            test_df = self.test_data
        else:
            raise ValueError(
                "Test set must be provided in time series custom mode: "
                "Please pass in scores(test_data=...), or provide test_data when initializing TimeSeriesML."
            )

        test_metrics = self._eval_on_test(test_df)

        result_df = pd.DataFrame(
            [train_metrics, test_metrics],
            index=["train", "test"],
        )
        return self._filter_metrics_columns(result_df, metrics)

    def _eval_train_only(
        self,
        metrics: Union[str, List[str]],
        external_test_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Time series train-only mode."""
        train_metrics = self._get_train_metrics()

        rows: List[Dict[str, float]] = [train_metrics]
        indices: List[str] = ["train"]

        test_df: Optional[pd.DataFrame] = None
        if external_test_data is not None:
            test_df = external_test_data
        elif self.test_data is not None:
            test_df = self.test_data

        if test_df is not None:
            test_metrics = self._eval_on_test(test_df)
            rows.append(test_metrics)
            indices.append("test")

        result_df = pd.DataFrame(rows, index=indices)
        return self._filter_metrics_columns(result_df, metrics)


# =============================================================================
# Clustering Strategy
# =============================================================================


class ClusteringScoresStrategy(ScoresStrategy):
    """Evaluation strategy for clustering tasks."""

    def evaluate(
        self,
        mode: str,
        metrics: Union[str, List[str]],
        external_test_data: Optional[pd.DataFrame],
        train_size: float,
        fold: int,
    ) -> pd.DataFrame:
        """Execute clustering evaluation (only supports train-only)."""
        if mode not in {"auto", "train-only"}:
            raise ValueError("Clustering only supports 'auto' and 'train-only' modes")

        return self._eval_train_only(metrics, external_test_data)

    def _eval_train_only(
        self,
        metrics: Union[str, List[str]],
        external_test_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Train-only mode evaluation."""
        if self.current_model is None:
            raise ValueError("No evaluable clustering model currently available, please create a model first")

        # Get feature columns and preprocessing pipeline
        try:
            X_cfg = self.experiment.get_config("X")
        except Exception as exc:
            raise ValueError("Cannot read clustering feature matrix X from experiment configuration") from exc

        feature_cols = list(X_cfg.columns)
        if not feature_cols:
            raise ValueError("No feature columns detected for clustering task, cannot calculate metrics")

        rows: List[Dict[str, float]] = []
        indices: List[str] = []

        # Clustering quality on training set
        train_metrics = self._eval_on_data(self.data, feature_cols)
        rows.append(train_metrics)
        indices.append("train")

        # Test set (optional)
        test_df: Optional[pd.DataFrame] = None
        if external_test_data is not None:
            test_df = external_test_data
        elif self.test_data is not None:
            test_df = self.test_data

        if test_df is not None:
            missing_cols = [col for col in feature_cols if col not in test_df.columns]
            if missing_cols:
                raise ValueError(
                    f"Clustering task test set missing the following feature columns, cannot perform evaluation: {missing_cols}"
                )
            test_metrics = self._eval_on_data(test_df, feature_cols)
            rows.append(test_metrics)
            indices.append("test")

        result_df = pd.DataFrame(rows, index=indices)
        return self._filter_metrics_columns(result_df, metrics)

    def _eval_on_data(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
    ) -> Dict[str, float]:
        """Calculate internal clustering metrics on given dataset."""
        if self.current_model is None:
            raise ValueError("No evaluable clustering model currently available")

        X = data[feature_cols]

        try:
            pipeline = self.experiment.get_config("pipeline")
        except Exception as exc:
            raise ValueError("Cannot get clustering preprocessing pipeline from experiment configuration") from exc

        X_trans = pipeline.transform(X)

        model = self.current_model
        labels = model.predict(X_trans)

        metrics_containers = getattr(self.experiment, "_all_metrics", {})
        scores: Dict[str, float] = {}

        for key, container in metrics_containers.items():
            if getattr(container, "needs_ground_truth", False):
                continue
            score_func = getattr(container, "score_func", None)
            if score_func is None:
                continue
            try:
                value = score_func(X_trans, labels)
            except Exception:
                continue
            if isinstance(value, (int, float, np.floating)):
                display_name = getattr(container, "display_name", key)
                scores[display_name] = float(value)

        if not scores:
            raise ValueError("Current clustering task failed to calculate any internal metrics, please check model and data configuration")

        return scores


# =============================================================================
# Factory Function
# =============================================================================


def get_scores_strategy(automl_instance: "AutoMLBase", task_type: str) -> ScoresStrategy:
    """Get evaluation strategy for the given task type."""
    if task_type == "supervised":
        return SupervisedScoresStrategy(automl_instance)
    elif task_type == "time_series":
        return TimeSeriesScoresStrategy(automl_instance)
    elif task_type == "clustering":
        return ClusteringScoresStrategy(automl_instance)
    else:
        raise NotImplementedError(f"No strategy for task type: {task_type}")
