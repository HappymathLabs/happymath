"""
AutoML base classes.

Core base for the HappyMath AutoML framework: data loading, experiment setup,
model storage and evaluation utilities; all task wrappers derive from this.
"""

from __future__ import annotations

import inspect
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

os.environ["PYCARET_CUSTOM_LOGGING_PATH"] = os.devnull
os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "CRITICAL"
os.environ["CATBOOST_ALLOW_WRITING_FILES"] = "0"

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, train_test_split

# Import the global Chinese font configuration
try:
    from .. import zh_font_available, zh_font_paths
except ImportError:
    zh_font_available = []
    zh_font_paths = {}

DataLike = Union[str, pd.DataFrame, np.ndarray, Tuple[np.ndarray, np.ndarray]]
TargetLike = Union[str, int, None]


@dataclass
class StoredModel:
    """Simple container to store model-related information."""

    model: Any
    metrics: Dict[str, Any]
    name: str
    extra: Dict[str, Any]
    timestamp: pd.Timestamp


class AutoMLBase:
    """
    HappyMath AutoML base class.

    Encapsulates the PyCaret experiment lifecycle and provides unified data loading, metric handling, and model management.
    """

    primary_metric: Optional[str] = None
    _catboost_patched: bool = False

    def __init__(
        self,
        data: DataLike,
        target: TargetLike = None,
        test_data: Optional[DataLike] = None,
        primary_metric: Optional[str] = None,
        **setup_kwargs: Any,
    ) -> None:
        # 预先禁用CatBoost产生本地文件夹（catboost_info等）
        self._silence_catboost_artifacts()
        # Data loading and validation
        self.data, normalized_target = self._load_data(data, target)
        self.target = normalized_target
        self._validate_data(self.data, self.target)

        # Test data processing
        if test_data is not None:
            self.test_data, _ = self._load_data(test_data, target=None)
        else:
            self.test_data = None

        # Common attribute initialization
        self.primary_metric = primary_metric or self.primary_metric
        self.setup_kwargs = setup_kwargs
        self.experiment = None
        self.models: Dict[str, StoredModel] = {}
        self.current_model: Optional[Any] = None
        self.is_setup = False
        self.results = None
        self.verbose = getattr(self, "verbose", False)

        # Auto-execute experiment initialization
        self._setup_experiment(**setup_kwargs)

    def _silence_catboost_artifacts(self) -> None:
        """通过猴子补丁为CatBoost默认加上allow_writing_files=False，避免生成日志目录。"""
        if AutoMLBase._catboost_patched:
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
                # 只设置 allow_writing_files 参数来禁用文件写入
                # 不设置 verbose/logging_level 因为 PyCaret 已经处理了这些
                if "allow_writing_files" in param_names:
                    kwargs.setdefault("allow_writing_files", False)
                return original_init(self, *args, **kwargs)

            wrapped.__signature__ = sig  # 保留签名便于反射
            cls.__init__ = wrapped  # type: ignore

        # 只补丁实际的分类器和回归器类
        for name in ("CatBoostRegressor", "CatBoostClassifier"):
            model_cls = getattr(catboost, name, None)
            if model_cls is not None:
                _patch_init(model_cls)

        AutoMLBase._catboost_patched = True

    # ------------------------------------------------------------------
    # Data-related tools
    # ------------------------------------------------------------------
    def _is_sklearn_bunch(self, data: Any) -> bool:
        """Check if data is a sklearn Bunch object (from load_*, fetch_* datasets)."""
        return (
            hasattr(data, 'data') and 
            hasattr(data, 'target') and 
            hasattr(data, 'feature_names')
        )
    
    def _handle_sklearn_bunch(self, data: Any, target: TargetLike) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Handle sklearn Bunch objects (load_*, fetch_* datasets).
        
        Args:
            data: sklearn Bunch object with .data, .target, .feature_names attributes
            target: None, string column name, integer index, or tuple index
            
        Returns:
            Tuple of (DataFrame, target_column_name)
        """
        # Extract features and target from Bunch object
        features_data = data.data
        target_data = data.target
        
        # Convert to arrays if needed
        if hasattr(features_data, 'values'):
            features_data = features_data.values
        if hasattr(target_data, 'values'):
            target_data = target_data.values
            
        # Create feature columns names
        if hasattr(data, 'feature_names') and data.feature_names:
            feature_columns = list(data.feature_names)
        else:
            feature_columns = [f"feature_{idx}" for idx in range(features_data.shape[1])]
        
        # Create DataFrame with features
        df = pd.DataFrame(features_data, columns=feature_columns)
        
        # Add target column
        if target is None:
            # Default target column name
            target_column_name = "target"
            df[target_column_name] = target_data
            return df, target_column_name
        elif isinstance(target, str):
            # Use provided column name
            target_column_name = target
            df[target_column_name] = target_data
            return df, target_column_name
        elif isinstance(target, int):
            # Handle integer target (special case: -1 means use default target)
            if target == -1:
                target_column_name = "target"
                df[target_column_name] = target_data
                return df, target_column_name
            elif 0 <= target < len(feature_columns):
                # Use specified feature column as target
                target_column_name = feature_columns[target]
                df.rename(columns={target_column_name: "target"}, inplace=True)
                return df, "target"
            else:
                raise ValueError(f"Target column index {target} out of range for {len(feature_columns)} features")
        else:
            raise TypeError("Target must be None, string column name, or integer index")
    
    def _load_data(
        self,
        data: DataLike,
        target: TargetLike,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Normalize various input data formats into a DataFrame and handle target column.
        Automatically handles sklearn datasets (Bunch objects, tuples, etc.).
        """
        # Auto-detect sklearn Bunch objects (load_*, fetch_* datasets)
        if self._is_sklearn_bunch(data):
            return self._handle_sklearn_bunch(data, target)
        
        # Handle regular data types
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
            # Target column handling in array/tuple mode is processed in conversion function
            return self._convert_array_to_frame(data, target)
        else:
            raise TypeError("data must be a file path, DataFrame, NumPy array, or tuple of (features, target)")

        # Continue with target column handling...
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

    def _convert_array_to_frame(
        self,
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
        # Convert single array + array/Series target to tuple mode for unified processing
        if isinstance(data, np.ndarray) and not isinstance(data, tuple):
            if isinstance(target, (np.ndarray, pd.Series)):
                target_array = np.asarray(target) if isinstance(target, pd.Series) else target
                data = (data, target_array)
                target = None

        # Handle tuple input: (features, target)
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

        # Handle single array input
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

    def _validate_data(self, data: pd.DataFrame, target: Optional[str]) -> None:
        """Basic data validation to ensure target exists and no duplicate columns."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("internal data must be a pandas.DataFrame")

        if data.columns.duplicated().any():
            raise ValueError("duplicate column names found; please resolve them first")

        if target is not None and target not in data.columns:
            raise ValueError("specified target column not found in data")

    # ------------------------------------------------------------------
    # Metric-related tools
    # ------------------------------------------------------------------
    def _get_metric_direction(self, metric: str) -> str:
        """Determine metric optimization direction from common names."""
        lower_better = {
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
        higher_better = {
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

        if metric in lower_better:
            return "lower_better"
        if metric in higher_better:
            return "higher_better"

        print(f"Warning: unknown metric '{metric}', defaulting to higher-is-better")
        return "higher_better"

    def _is_better_score(self, new_score: float, current_best: Optional[float]) -> bool:
        """Determine whether a new score is better based on metric direction."""
        if current_best is None:
            return True

        direction = self._get_metric_direction(self.primary_metric)
        if direction == "higher_better":
            return new_score > current_best
        return new_score < current_best

    def _safe_get_model_name(self, model: Any) -> str:
        """Get a readable model name, preferring experiment-provided helpers."""
        if self.experiment and hasattr(self.experiment, "_get_model_name"):
            try:
                return self.experiment._get_model_name(model)
            except Exception:
                pass
        return getattr(model, "__class__", model).__name__

    def _extract_metrics_from_results(
        self,
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

    def _filter_kwargs_for(self, func: Any, base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter unsupported kwargs by function signature."""
        params = inspect.signature(func).parameters
        return {key: value for key, value in base_kwargs.items() if key in params}

    def _get_chinese_font(self) -> str:
        """获取可用的中文字体名称。"""
        if zh_font_available and isinstance(zh_font_available, list) and len(zh_font_available) > 0:
            return zh_font_available[0]
        return "DejaVu Sans"

    def _get_font_properties(self):
        """
        获取用于中文显示的 FontProperties 对象。
        直接使用 __init__.py 中已检测到的字体路径。
        """
        from matplotlib.font_manager import FontProperties

        chinese_font = self._get_chinese_font()

        # 直接使用 __init__.py 中保存的字体路径
        if zh_font_paths and chinese_font in zh_font_paths:
            return FontProperties(fname=zh_font_paths[chinese_font])

        # 回退到使用字体名称
        return FontProperties(family=chinese_font)

    def _apply_chinese_font_to_figure(
        self,
        fig: Any,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        legend_title: Optional[str] = None,
        legend_labels: Optional[List[str]] = None,
        font_sizes: Optional[Dict[str, Union[int, float]]] = None,
    ) -> None:
        """
        将中文字体应用到 matplotlib Figure 的所有文本元素。
        在 plt.show() 或 plt.savefig() 之前调用。
        """
        import warnings

        fonts = font_sizes or {}

        # 使用智能字体获取方法
        font_prop = self._get_font_properties()

        def apply_font_to_text(text_obj, new_text=None, fontsize=None):
            """辅助函数：将中文字体应用到单个文本对象"""
            if text_obj is None:
                return
            if new_text is not None:
                text_obj.set_text(new_text)
            text_obj.set_fontproperties(font_prop)
            if fontsize is not None:
                text_obj.set_fontsize(fontsize)

        for ax in fig.get_axes():
            # 标题
            if ax.title:
                apply_font_to_text(ax.title, title, fonts.get("title"))

            # X轴标签
            if ax.xaxis.label:
                apply_font_to_text(ax.xaxis.label, xlabel, fonts.get("xlabel"))

            # Y轴标签
            if ax.yaxis.label:
                apply_font_to_text(ax.yaxis.label, ylabel, fonts.get("ylabel"))

            # 刻度标签
            tick_size_x = fonts.get("tick") or fonts.get("xtick")
            for label in ax.get_xticklabels():
                apply_font_to_text(label, fontsize=tick_size_x)

            tick_size_y = fonts.get("tick") or fonts.get("ytick")
            for label in ax.get_yticklabels():
                apply_font_to_text(label, fontsize=tick_size_y)

            # 图例
            legend = ax.get_legend()
            if legend:
                if legend_title is not None:
                    legend.set_title(legend_title)
                if legend.get_title():
                    legend.get_title().set_fontproperties(font_prop)
                    if fonts.get("legend_title"):
                        legend.get_title().set_fontsize(fonts["legend_title"])

                texts = legend.get_texts()
                if legend_labels is not None:
                    if len(legend_labels) != len(texts):
                        warnings.warn(
                            "legend_labels length mismatches legend entries; will truncate to the minimal length",
                            UserWarning,
                        )
                    for text_obj, lbl in zip(texts, legend_labels):
                        text_obj.set_text(lbl)
                for text_obj in texts:
                    text_obj.set_fontproperties(font_prop)
                    if fonts.get("legend_label"):
                        text_obj.set_fontsize(fonts["legend_label"])

            # 所有其他文本注释
            for text in ax.texts:
                text.set_fontproperties(font_prop)

        # Figure 级别的 suptitle
        if hasattr(fig, "_suptitle") and fig._suptitle:
            fig._suptitle.set_fontproperties(font_prop)

    @contextmanager
    def _chinese_font_context(
        self,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        legend_title: Optional[str] = None,
        legend_labels: Optional[List[str]] = None,
        font_sizes: Optional[Dict[str, Union[int, float]]] = None,
    ):
        """
        上下文管理器：拦截 plt.show() 和 plt.savefig()，在执行前应用中文字体。
        """
        import matplotlib.pyplot as plt

        original_show = plt.show
        original_savefig = plt.savefig
        original_figure_savefig = plt.Figure.savefig

        def intercepted_show(*args, **kwargs):
            fig = plt.gcf()
            self._apply_chinese_font_to_figure(
                fig, title, xlabel, ylabel, legend_title, legend_labels, font_sizes
            )
            return original_show(*args, **kwargs)

        def intercepted_savefig(fname, *args, **kwargs):
            fig = plt.gcf()
            self._apply_chinese_font_to_figure(
                fig, title, xlabel, ylabel, legend_title, legend_labels, font_sizes
            )
            return original_savefig(fname, *args, **kwargs)

        def intercepted_figure_savefig(self_fig, fname, *args, **kwargs):
            self._apply_chinese_font_to_figure(
                self_fig, title, xlabel, ylabel, legend_title, legend_labels, font_sizes
            )
            return original_figure_savefig(self_fig, fname, *args, **kwargs)

        try:
            plt.show = intercepted_show
            plt.savefig = intercepted_savefig
            plt.Figure.savefig = intercepted_figure_savefig
            yield
        finally:
            plt.show = original_show
            plt.savefig = original_savefig
            plt.Figure.savefig = original_figure_savefig

    # ------------------------------------------------------------------
    # Model storage and management
    # ------------------------------------------------------------------
    def _store_model_with_metrics(
        self,
        model: Any,
        model_name: str,
        results_df: Optional[Any] = None,
        model_label: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record model object with metrics for later reference."""
        metrics: Dict[str, Any] = {}

        if results_df is None:
            try:
                results_df = self.experiment.pull()
            except Exception:
                results_df = None

        if results_df is not None:
            metrics = self._extract_metrics_from_results(results_df, model_label)

        # Assemble additional information and preserve complete cross-validation results for time series tasks
        extra: Dict[str, Any] = dict(additional_info) if additional_info is not None else {}

        usecase = getattr(self.experiment, "_ml_usecase", None)
        usecase_str = str(usecase) if usecase is not None else ""
        if "TIME_SERIES" in usecase_str and isinstance(results_df, pd.DataFrame):
            # Time series create_model steps' pull usually returns cross-validation results with cutoff column and Mean/SD rows
            has_cutoff_col = "cutoff" in results_df.columns
            has_mean_row = any(str(idx).lower() == "mean" for idx in results_df.index)
            if has_cutoff_col and has_mean_row:
                # Save a complete copy for reuse in scores (kfold etc.)
                extra.setdefault("ts_cv_results", results_df.copy())

        info = StoredModel(
            model=model,
            metrics=metrics,
            name=model_name,
            extra=extra,
            timestamp=pd.Timestamp.now(),
        )
        self.models[model_name] = info

    def get_best_model(self) -> Tuple[Any, Dict[str, Any]]:
        """Return the best model based on primary_metric."""
        if not self.models:
            raise ValueError("No comparable models; please create or compare models first")

        best_name = None
        best_info: Optional[StoredModel] = None
        best_score: Optional[float] = None

        for name, info in self.models.items():
            score = info.metrics.get(self.primary_metric)
            if score is None:
                continue
            if self._is_better_score(score, best_score):
                best_score = score
                best_name = name
                best_info = info

        if best_info is None:
            fallback = list(self.models.values())[-1]
            print(
                f"Warning: No model contains the primary metric '{self.primary_metric}', will return the most recent model '{fallback.name}'"
            )
            self.current_model = fallback.model
            return fallback.model, fallback.metrics

        self.current_model = best_info.model
        return best_info.model, best_info.metrics

    # ------------------------------------------------------------------
    # Core training and evaluation interfaces
    # ------------------------------------------------------------------
    def compare(
        self,
        include: Optional[List[Any]] = None,
        exclude: Optional[List[str]] = None,
        sort: Optional[str] = None,
        budget_time: Optional[float] = None,
        verbose: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """Compare supported models and select the best performer."""
        self._ensure_setup()
        verbose_flag = self.verbose if verbose is None else verbose
        metric = sort or self.primary_metric

        best_model = self.experiment.compare_models(
            include=include,
            exclude=exclude,
            sort=metric,
            budget_time=budget_time,
            verbose=verbose_flag,
            n_select=1,
            turbo=False,
            **kwargs,
        )

        results = self.experiment.pull()
        self.results = results
        label = self._safe_get_model_name(best_model)
        self._store_model_with_metrics(
            best_model,
            model_name="compare_best",
            results_df=results,
            model_label=label,
        )
        self.current_model = best_model
        return best_model

    def create(
        self,
        estimator: Any,
        return_train_score: bool = False,
        verbose: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a model with the specified algorithm."""
        self._ensure_setup()
        verbose_flag = self.verbose if verbose is None else verbose

        model = self.experiment.create_model(
            estimator=estimator,
            return_train_score=return_train_score,
            verbose=verbose_flag,
            **kwargs,
        )

        results = self.experiment.pull()
        self.results = results
        label = self._safe_get_model_name(model)
        self._store_model_with_metrics(
            model,
            model_name=f"create_{label}",
            results_df=results,
            model_label=label,
        )
        if self.current_model is None:
            self.current_model = model
        return model

    def tune(
        self,
        estimator: Optional[Any] = None,
        n_iter: int = 10,
        custom_grid: Optional[Dict[str, List[Any]]] = None,
        optimize: Optional[str] = None,
        verbose: Optional[bool] = None,
        tuner_verbose: Union[int, bool] = True,
        **kwargs: Any,
    ) -> Any:
        """Tune hyperparameters for the current or a specified model."""
        self._ensure_setup()

        base_model = estimator or self.current_model
        if base_model is None:
            raise ValueError("No model to tune; please run compare or create first")

        metric = optimize or self.primary_metric
        verbose_flag = self.verbose if verbose is None else verbose

        tuned_model = self.experiment.tune_model(
            estimator=base_model,
            n_iter=n_iter,
            custom_grid=custom_grid,
            optimize=metric,
            verbose=verbose_flag,
            tuner_verbose=tuner_verbose,
            choose_better=True,
            **kwargs,
        )

        results = self.experiment.pull()
        self.results = results
        label = self._safe_get_model_name(tuned_model)
        self._store_model_with_metrics(
            tuned_model,
            model_name="tuned",
            results_df=results,
            model_label=label,
        )
        self.current_model = tuned_model
        return tuned_model

    def ensemble(
        self,
        estimator: Optional[Any] = None,
        method: str = "Bagging",
        n_estimators: int = 10,
        optimize: Optional[str] = None,
        verbose: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """Apply bagging/boosting ensembling to the model."""
        self._ensure_setup()
        base_model = estimator or self.current_model
        if base_model is None:
            raise ValueError("No model available for ensembling")

        metric = optimize or self.primary_metric
        verbose_flag = self.verbose if verbose is None else verbose

        ensemble_call = {
            "estimator": base_model,
            "method": method,
            "n_estimators": n_estimators,
            "optimize": metric,
            "choose_better": True,
            "verbose": verbose_flag,
        }
        ensemble_call.update(kwargs)
        filtered = self._filter_kwargs_for(self.experiment.ensemble_model, ensemble_call)

        ensemble_model = self.experiment.ensemble_model(**filtered)

        results = self.experiment.pull()
        self.results = results
        label = self._safe_get_model_name(ensemble_model)
        self._store_model_with_metrics(
            ensemble_model,
            model_name=f"ensemble_{method.lower()}",
            results_df=results,
            model_label=label,
            additional_info={
                "ensemble_method": method,
                "n_estimators": n_estimators,
            },
        )
        self.current_model = ensemble_model
        return ensemble_model

    def blend(
        self,
        estimator_list: Optional[List[Any]] = None,
        optimize: Optional[str] = None,
        method: str = "auto",
        weights: Optional[List[float]] = None,
        verbose: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """Blend multiple models via voting/averaging."""
        self._ensure_setup()

        if estimator_list is None:
            if len(self.models) < 2:
                raise ValueError("At least two base models are required to blend")
            estimator_list = [info.model for info in self.models.values()]

        metric = optimize or self.primary_metric
        verbose_flag = self.verbose if verbose is None else verbose

        blend_call = {
            "estimator_list": estimator_list,
            "method": method,
            "weights": weights,
            "optimize": metric,
            "choose_better": True,
            "verbose": verbose_flag,
        }
        blend_call.update(kwargs)
        filtered = self._filter_kwargs_for(self.experiment.blend_models, blend_call)

        blended = self.experiment.blend_models(**filtered)

        results = self.experiment.pull()
        self.results = results
        label = self._safe_get_model_name(blended)
        self._store_model_with_metrics(
            blended,
            model_name="blended",
            results_df=results,
            model_label=label,
            additional_info={
                "blend_method": method,
                "n_models": len(estimator_list),
            },
        )
        self.current_model = blended
        return blended

    def stack(
        self,
        estimator_list: Optional[List[Any]] = None,
        meta_model: Optional[Any] = None,
        meta_model_fold: Optional[int] = 5,
        method: str = "auto",
        restack: bool = False,
        optimize: Optional[str] = None,
        verbose: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """Stack models to build a two-layer ensemble."""
        self._ensure_setup()

        if estimator_list is None:
            if len(self.models) < 2:
                raise ValueError("At least two base models are required to stack")
            estimator_list = [info.model for info in self.models.values()]

        metric = optimize or self.primary_metric
        verbose_flag = self.verbose if verbose is None else verbose

        stack_call = {
            "estimator_list": estimator_list,
            "meta_model": meta_model,
            "meta_model_fold": meta_model_fold,
            "method": method,
            "restack": restack,
            "optimize": metric,
            "choose_better": True,
            "verbose": verbose_flag,
        }
        stack_call.update(kwargs)
        filtered = self._filter_kwargs_for(self.experiment.stack_models, stack_call)

        stacked = self.experiment.stack_models(**filtered)

        results = self.experiment.pull()
        self.results = results
        label = self._safe_get_model_name(stacked)
        self._store_model_with_metrics(
            stacked,
            model_name="stacked",
            results_df=results,
            model_label=label,
            additional_info={
                "meta_model": self._safe_get_model_name(meta_model)
                if meta_model
                else "LogisticRegression",
                "n_base_models": len(estimator_list),
            },
        )
        self.current_model = stacked
        return stacked

    def plot(
        self,
        estimator: Optional[Any] = None,
        plot_type: str = "auc",
        scale: float = 1.0,
        save: bool = False,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        legend_title: Optional[str] = None,
        legend_labels: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 6),
        plot_kwargs: Optional[Dict[str, Any]] = None,
        font_sizes: Optional[Dict[str, Union[int, float]]] = None,
        verbose: Optional[bool] = None,
    ) -> Optional[str]:
        """Call PyCaret plotting with friendly default titles."""
        import warnings

        self._ensure_setup()
        estimator = estimator or self.current_model
        if estimator is None:
            raise ValueError("No model available for plotting")

        verbose_flag = self.verbose if verbose is None else verbose

        # 构建传递给 PyCaret 的参数（仅保留 figsize 等布局参数）
        # 标题、坐标轴标签等通过 _chinese_font_context 后处理设置
        final_kwargs = dict(plot_kwargs or {})
        final_kwargs.setdefault("figsize", figsize)

        plot_call = {
            "estimator": estimator,
            "plot": plot_type,
            "scale": scale,
            "save": save,
            "verbose": verbose_flag,
            "plot_kwargs": final_kwargs,
            "fig_kwargs": final_kwargs,
        }
        filtered_call = self._filter_kwargs_for(self.experiment.plot_model, plot_call)

        # 使用后处理上下文管理器设置中文字体和自定义文本
        customization_context = self._chinese_font_context(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            legend_title=legend_title,
            legend_labels=legend_labels,
            font_sizes=font_sizes,
        )

        with customization_context:
            try:
                result = self.experiment.plot_model(**filtered_call)
                # 处理 Plotly 图表的中文字体
                result = self._apply_plotly_chinese_font(
                    result, title=title, xlabel=xlabel, ylabel=ylabel
                )
                return result
            except Exception as exc:
                warnings.warn(f"Plotting with custom parameters failed, falling back to default. Error: {exc}")
                fallback = {
                    "estimator": estimator,
                    "plot": plot_type,
                    "scale": scale,
                    "save": save,
                    "verbose": verbose_flag,
                }
                fallback_filtered = self._filter_kwargs_for(
                    self.experiment.plot_model, fallback
                )
                return self.experiment.plot_model(**fallback_filtered)

    def _apply_plotly_chinese_font(
        self,
        fig: Any,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ) -> Any:
        """
        检测并处理 Plotly 图表的中文字体设置。

        如果返回的 fig 是 Plotly Figure 对象，则设置其字体为 zh_font_available 中的中文字体。
        """
        # 检测是否为 Plotly Figure
        try:
            import plotly.graph_objects as go
        except ImportError:
            # 未安装 plotly，直接返回原对象
            return fig

        if not isinstance(fig, go.Figure):
            # 不是 Plotly Figure，直接返回
            return fig

        # 获取中文字体
        if zh_font_available and isinstance(zh_font_available, list) and len(zh_font_available) > 0:
            chinese_font = zh_font_available[0]
        else:
            chinese_font = "Arial"

        # 更新全局字体设置
        fig.update_layout(
            font=dict(family=chinese_font),
        )

        # 如果用户指定了标题，更新标题
        if title is not None:
            fig.update_layout(
                title=dict(text=title, font=dict(family=chinese_font)),
            )

        # 如果用户指定了坐标轴标签，更新坐标轴
        if xlabel is not None:
            fig.update_xaxes(title=dict(text=xlabel, font=dict(family=chinese_font)))
        if ylabel is not None:
            fig.update_yaxes(title=dict(text=ylabel, font=dict(family=chinese_font)))

        return fig

    def predict(
        self,
        estimator: Optional[Any] = None,
        data: Optional[pd.DataFrame] = None,
        raw_score: bool = False,
        verbose: Optional[bool] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Predict with the specified model or the test set by default."""
        self._ensure_setup()
        predictor = estimator or self.current_model
        if predictor is None:
            raise ValueError("No model available for prediction")

        data_to_use = data if data is not None else self.test_data
        verbose_flag = self.verbose if verbose is None else verbose

        predict_fn = self.experiment.predict_model
        signature = inspect.signature(predict_fn)
        call_kwargs = {
            "estimator": predictor,
            "data": data_to_use,
            "verbose": verbose_flag,
            **kwargs,
        }
        if "raw_score" in signature.parameters:
            call_kwargs["raw_score"] = raw_score
        elif raw_score:
            print("Warning: current task does not support raw_score; it will be ignored")

        return predict_fn(**call_kwargs)

    def finalize(self, estimator: Optional[Any] = None) -> Any:
        """Finalize model by training on the full dataset."""
        self._ensure_setup()
        target_model = estimator or self.current_model
        if target_model is None:
            raise ValueError("No model available to finalize")

        final_model = self.experiment.finalize_model(target_model)
        results = self.experiment.pull()
        self.results = results
        label = self._safe_get_model_name(final_model)
        self._store_model_with_metrics(
            final_model,
            model_name="final",
            results_df=results,
            model_label=label,
        )
        self.current_model = final_model
        return final_model

    def evaluate(self, estimator: Optional[Any] = None) -> None:
        """Start interactive evaluation UI."""
        self._ensure_setup()
        target_model = estimator or self.current_model
        if target_model is None:
            raise ValueError("No model available for evaluation")
        self.experiment.evaluate_model(target_model)

    # ------------------------------------------------------------------
    # External helper interfaces
    # ------------------------------------------------------------------
    def get_models(self) -> Iterable[str]:
        """Return the list of stored model names."""
        return list(self.models.keys())

    def get_metrics(self) -> Any:
        """Return the list of supported metrics."""
        self._ensure_setup()
        if hasattr(self.experiment, "get_metrics"):
            return self.experiment.get_metrics()
        if hasattr(self.experiment, "_all_metrics"):
            containers = getattr(self.experiment, "_all_metrics")
            rows = []
            for key, container in containers.items():
                display = getattr(container, "display_name", key)
                rows.append({"ID": key, "Display Name": display})
            return pd.DataFrame(rows)
        raise AttributeError("Current experiment does not support get_metrics")

    def get_results(self) -> pd.DataFrame:
        """Get the latest results table."""
        self._ensure_setup()
        if self.results is not None:
            return self.results
        pulled = self.experiment.pull()
        if pulled is None:
            raise ValueError("No results table available")
        return pulled

    def get_leaderboard(self) -> pd.DataFrame:
        """Get the model leaderboard."""
        self._ensure_setup()
        if hasattr(self.experiment, "get_leaderboard"):
            return self.experiment.get_leaderboard()
        if self.results is not None and not self.results.empty:
            return self.results
        pulled = self.experiment.pull()
        if pulled is not None and not pulled.empty:
            return pulled
            raise ValueError("No leaderboard data available")

    def save(self, model_name: str, model: Optional[Any] = None) -> None:
        """Save the model to disk."""
        self._ensure_setup()
        model_to_save = model or self.current_model
        if model_to_save is None:
            raise ValueError("No model available to save")
        self.experiment.save_model(model_to_save, model_name)

    def load(self, model_name: str) -> Any:
        """Load a model from disk."""
        self._ensure_setup()
        return self.experiment.load_model(model_name)

    def get_config(self, key: Optional[str] = None) -> Any:
        """Read experiment configuration."""
        self._ensure_setup()
        return self.experiment.get_config(key)

    # ------------------------------------------------------------------
    # Unified evaluation interface
    # ------------------------------------------------------------------
    def scores(
        self,
        mode: str = "auto",
        metrics: Union[str, List[str]] = "all",
        test_data: Optional[DataLike] = None,
        train_size: Optional[float] = None,
        fold: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Evaluate model performance across different data split modes using the current model, returning a DataFrame.

        Parameters
        ----------
        mode:
            Evaluation mode, options:
            - ``auto`` (default):
              - If test set exists (``test_data`` in scores or ``self.test_data`` from initialization), equivalent to ``custom``;
              - Otherwise automatically selected based on sample size:
                * Supervised learning: n < 100 use ``leaveout``; 100 ≤ n ≤ 10000 use ``kfold``; n > 10000 use ``holdout``;
                * Time series: n ≤ 10000 use ``kfold``; n > 10000 use ``holdout``;
                * Clustering: forced to ``train-only``.
            - ``holdout``: Split train/test sets according to ``train_size`` ratio, evaluate each once;
            - ``kfold``: Cross-validation with ``fold`` folds:
              * Classification/regression: one row per fold, last row shows average results;
              * Time series: reuse PyCaret's time series backtesting results;
            - ``leaveout``: Leave-one-out evaluation only supported for supervised learning (not supported for time series);
            - ``custom``: Training set uses ``data`` from initialization, test set uses ``test_data`` parameter from scores,
              if empty falls back to ``self.test_data``, error if both are empty;
            - ``train-only``: No new splitting:
              * Always evaluate training set;
              * If test set exists (``test_data`` in scores or ``self.test_data``), also evaluate test set.
              * For clustering tasks, only supports ``auto`` and ``train-only``, where ``auto`` is equivalent to ``train-only``.
        metrics:
            Metric selection:
            - ``"all"``: Return all computable metrics for current task;
            - Single string: Keep only this metric (case-insensitive, e.g. ``"accuracy"`` is equivalent to ``"Accuracy"``);
            - String list: Keep only metrics appearing in the list.
        test_data:
            Custom test set, only used in ``custom`` or ``train-only`` modes.
            - For classification/regression tasks, test set must contain target column (consistent with ``target`` from initialization), otherwise metrics cannot be calculated;
            - For time series/clustering tasks, target column is optional.
        train_size:
            Training set proportion in holdout mode, range (0, 1). Resolution priority:
            1. If explicitly passed as non-None in ``scores``, use directly;
            2. Otherwise use ``train_size`` in ``self.setup_kwargs`` from initialization (if exists and non-None);
            3. Otherwise use internal default value 0.7.
        fold:
            Number of folds in kfold mode. Resolution priority:
            1. If explicitly passed as non-None in ``scores``, use directly;
            2. Otherwise use ``fold`` in ``self.setup_kwargs`` from initialization (if exists and non-None);
            3. Otherwise use internal default value 5.

        Returns
        ----------
        df : pandas.DataFrame
            - ``holdout``: Two rows, index is ``["train", "test"]``;
            - ``custom``: Two rows, index is ``["train", "test"]``;
            - ``train-only``: At least contains ``"train"`` row, if test set exists then also contains ``"test"`` row;
            - ``kfold``:
              * Supervised learning: ``fold`` rows show results per fold, last row ``"mean"`` shows average results;
              * Time series: Each ``cutoff`` corresponds to one row, last row shows average results;
            - ``leaveout`` (supervised learning only): Two rows, index is ``["train_mean", "test_mean"]``.
        """
        self._ensure_setup()
        if self.current_model is None:
            raise ValueError(
                "No evaluable model currently available, please create a model first using compare/create/tune/ensemble/blend/stack/finalize methods."
            )

        # Parse task type as internal unified label to avoid direct dependency on external enums
        usecase = getattr(self.experiment, "_ml_usecase", None)
        usecase_str = str(usecase) if usecase is not None else ""
        if "TIME_SERIES" in usecase_str:
            task_type = "time_series"
        elif "CLUSTERING" in usecase_str:
            task_type = "clustering"
        elif "CLASSIFICATION" in usecase_str or "REGRESSION" in usecase_str:
            task_type = "supervised"
        else:
            raise NotImplementedError("Current AutoML task type does not support scores() interface")

        # Parse external test set (if any), maintain consistency with initialization data specifications
        external_test_df: Optional[pd.DataFrame] = None
        if test_data is not None:
            # For supervised tasks, require custom test set to explicitly contain target column
            target_for_loading: Optional[TargetLike] = self.target if task_type != "clustering" else None
            external_test_df, _ = self._load_data(test_data, target=target_for_loading)

        # Parse train_size / fold with clear priority and internal default values
        effective_train_size = self._get_effective_train_size(train_size)
        effective_fold = self._get_effective_fold(fold)

        mode_normalized = (mode or "auto").lower()

        if task_type == "clustering":
            # Clustering tasks: only allow auto and train-only, where auto is equivalent to train-only
            if mode_normalized not in {"auto", "train-only"}:
                raise ValueError("Current task is clustering, scores only supports 'auto' and 'train-only' modes")
            resolved_mode = "train-only"
            return self._scores_clustering(
                mode=resolved_mode,
                metrics=metrics,
                external_test_data=external_test_df,
            )

        if task_type == "time_series":
            return self._scores_time_series(
                mode=mode_normalized,
                metrics=metrics,
                external_test_data=external_test_df,
                train_size=effective_train_size,
                fold=effective_fold,
            )

        # Supervised learning (classification / regression)
        return self._scores_supervised(
            mode=mode_normalized,
            metrics=metrics,
            external_test_data=external_test_df,
            train_size=effective_train_size,
            fold=effective_fold,
        )

    # ------------------------------------------------------------------
    # Helper tools
    # ------------------------------------------------------------------
    def _ensure_setup(self) -> None:
        """Ensure the experiment has been set up."""
        if not self.is_setup:
            raise RuntimeError("Please complete experiment setup first")

    def _get_effective_train_size(self, train_size: Optional[float]) -> float:
        """
        Parse train_size used in scores.

        Priority:
        1. train_size explicitly passed during scores call (non-None);
        2. train_size in self.setup_kwargs from initialization (non-None);
        3. Internal default value 0.7.
        """
        if train_size is not None:
            return float(train_size)

        from_kwargs = self.setup_kwargs.get("train_size", None)
        if from_kwargs is not None:
            try:
                return float(from_kwargs)
            except (TypeError, ValueError):
                pass

        return 0.7

    def _get_effective_fold(self, fold: Optional[int]) -> int:
        """
        Parse fold (cross-validation folds) used in scores.

        Priority:
        1. fold explicitly passed during scores call (non-None);
        2. fold in self.setup_kwargs from initialization (non-None);
        3. Internal default value 5.
        """
        if fold is not None:
            try:
                fold_int = int(fold)
            except (TypeError, ValueError):
                fold_int = 5
        else:
            from_kwargs = self.setup_kwargs.get("fold", None)
            if from_kwargs is not None:
                try:
                    fold_int = int(from_kwargs)
                except (TypeError, ValueError):
                    fold_int = 5
            else:
                fold_int = 5

        # At least 2 folds to avoid invalid configuration
        if fold_int < 2:
            fold_int = 2
        return fold_int

    def _get_random_state(self) -> Optional[int]:
        """
        Try to infer random seed used for data splitting.

        Priority:
        1. session_id in self.setup_kwargs from initialization;
        2. Common seed attribute on subclasses;
        3. Default None (handled by downstream functions).
        """
        from_kwargs = self.setup_kwargs.get("session_id", None)
        if from_kwargs is not None:
            return from_kwargs
        return getattr(self, "seed", None)

    def _filter_metrics_columns(self, df: pd.DataFrame, metrics: Union[str, List[str]]) -> pd.DataFrame:
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

        # For missing metrics, print one-time warning without interrupting main flow
        missing = requested - available_bases
        if missing:
            print(f"Warning: The following metrics are not found in current results and will be ignored: {sorted(missing)}")

        if not selected_cols:
            # If no matching columns, return original DataFrame to avoid empty table causing debugging difficulties
            return df
        return df[selected_cols]

    # ------------------------------------------------------------------
    # Task-specific scores implementations (classification/regression, time series, clustering)
    # ------------------------------------------------------------------

    def _scores_supervised(
        self,
        mode: str,
        metrics: Union[str, List[str]],
        external_test_data: Optional[pd.DataFrame],
        train_size: float,
        fold: int,
    ) -> pd.DataFrame:
        """
        Scores implementation for supervised learning (classification / regression).
        """
        allowed_modes = {"auto", "holdout", "kfold", "leaveout", "custom", "train-only"}

        if mode not in allowed_modes:
            raise ValueError(f"Unknown evaluation mode '{mode}', available options: {sorted(allowed_modes)}")

        has_any_test = external_test_data is not None or self.test_data is not None

        # auto mode automatically selects based on test set availability and sample size
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
            result = self._scores_supervised_holdout(
                metrics=metrics,
                external_test_data=external_test_data,
                train_size=train_size,
            )
        elif resolved_mode == "custom":
            result = self._scores_supervised_custom(
                metrics=metrics,
                external_test_data=external_test_data,
            )
        elif resolved_mode == "train-only":
            result = self._scores_supervised_train_only(
                metrics=metrics,
                external_test_data=external_test_data,
            )
        elif resolved_mode == "kfold":
            result = self._scores_supervised_kfold(
                metrics=metrics,
                fold=fold,
            )
        elif resolved_mode == "leaveout":
            result = self._scores_supervised_leaveout(metrics=metrics)
        else:
            # Theoretically should not reach here
            raise RuntimeError(f"Internal error: unhandled supervised learning evaluation mode '{resolved_mode}'")

        return result

    def _ensure_supervised_target(self) -> str:
        """
        Ensure current task is supervised learning and target column exists, return target column name.
        """
        if self.target is None:
            raise ValueError("Current task has no target column configured, cannot evaluate as supervised learning")
        if self.target not in self.data.columns:
            raise ValueError(f"Target column '{self.target}' does not exist in internal data")
        return self.target

    def _eval_supervised_split(self, data_split: pd.DataFrame) -> Dict[str, float]:
        """
        Call PyCaret's predict_model on given data subset and extract numeric metrics.
        """
        self._ensure_setup()
        if self.current_model is None:
            raise ValueError("No evaluable model currently available, please create or train a model first")

        # For supervised learning, require subset to contain target column, this constraint ensured by upstream splitting logic
        self.experiment.predict_model(
            estimator=self.current_model,
            data=data_split,
            verbose=self.verbose,
        )
        results_df = self.experiment.pull()
        metrics_dict = self._extract_metrics_from_results(results_df, model_label=None)

        numeric_metrics: Dict[str, float] = {}
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float, np.floating)):
                numeric_metrics[key] = float(value)
        return numeric_metrics

    def _scores_supervised_holdout(
        self,
        metrics: Union[str, List[str]],
        external_test_data: Optional[pd.DataFrame],
        train_size: float,
    ) -> pd.DataFrame:
        """Supervised learning holdout mode: split train/test sets once according to train_size."""
        target_col = self._ensure_supervised_target()
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
            # When sample size too small or class imbalance too severe causing stratification failure, fall back to regular splitting
            train_df, test_df = train_test_split(
                df,
                train_size=train_size,
                random_state=random_state,
            )

        # If user explicitly passes external_test_data, override the split test set
        if external_test_data is not None:
            if target_col not in external_test_data.columns:
                raise ValueError(
                    f"Custom test set missing target column '{target_col}', cannot calculate supervised task evaluation metrics"
                )
            test_df = external_test_data

        train_metrics = self._eval_supervised_split(train_df)
        test_metrics = self._eval_supervised_split(test_df)

        result_df = pd.DataFrame(
            [train_metrics, test_metrics],
            index=["train", "test"],
        )
        return self._filter_metrics_columns(result_df, metrics)

    def _scores_supervised_custom(
        self,
        metrics: Union[str, List[str]],
        external_test_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Supervised learning custom mode: training set is initialization data, test set provided by parameters/attributes."""
        target_col = self._ensure_supervised_target()
        train_df = self.data

        # Priority: use test_data from scores call first, then use self.test_data from initialization
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

        train_metrics = self._eval_supervised_split(train_df)
        test_metrics = self._eval_supervised_split(test_df)

        result_df = pd.DataFrame(
            [train_metrics, test_metrics],
            index=["train", "test"],
        )
        return self._filter_metrics_columns(result_df, metrics)

    def _scores_supervised_train_only(
        self,
        metrics: Union[str, List[str]],
        external_test_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Supervised learning train-only mode:
        Always evaluate training set, if test set exists (provided by parameters or initialization) also evaluate test set.
        """
        target_col = self._ensure_supervised_target()
        train_df = self.data

        rows: List[Dict[str, float]] = []
        indices: List[str] = []

        train_metrics = self._eval_supervised_split(train_df)
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
            test_metrics = self._eval_supervised_split(test_df)
            rows.append(test_metrics)
            indices.append("test")

        result_df = pd.DataFrame(rows, index=indices)
        return self._filter_metrics_columns(result_df, metrics)

    def _scores_supervised_kfold(
        self,
        metrics: Union[str, List[str]],
        fold: int,
    ) -> pd.DataFrame:
        """
        Supervised learning kfold mode: evaluate train/test subsets for each fold and provide mean.

        Each fold row contains both training and testing metrics, column names use
        ``<metric_name>_train`` / ``<metric_name>_test`` format; last row ``mean`` is column average.
        """
        target_col = self._ensure_supervised_target()
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

        # Used for calculating column means
        sum_accumulator: Dict[str, float] = {}
        n_folds_actual = 0

        for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            train_metrics = self._eval_supervised_split(train_df)
            test_metrics = self._eval_supervised_split(test_df)

            row: Dict[str, float] = {}
            # Unify key set to avoid some metrics only appearing in training or testing
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

    def _scores_supervised_leaveout(
        self,
        metrics: Union[str, List[str]],
    ) -> pd.DataFrame:
        """
        Supervised learning leaveout mode: leave-one-out cross-validation, returns train/test average metrics.

        Note: This mode has very high overhead when sample size is large, so in auto mode it's only automatically selected when n < 100.
        """
        target_col = self._ensure_supervised_target()
        df = self.data
        n_samples = len(df)
        if n_samples <= 1:
            raise ValueError("Leave-one-out cross-validation requires at least 2 samples")

        loo = LeaveOneOut()

        # Use sum and divide by rounds approach to accumulate average
        train_sum: Dict[str, float] = {}
        test_sum: Dict[str, float] = {}
        n_rounds = 0

        for train_idx, test_idx in loo.split(df):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            train_metrics = self._eval_supervised_split(train_df)
            test_metrics = self._eval_supervised_split(test_df)

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

    def _scores_time_series(
        self,
        mode: str,
        metrics: Union[str, List[str]],
        external_test_data: Optional[pd.DataFrame],
        train_size: float,
        fold: int,
    ) -> pd.DataFrame:
        """
        Scores implementation for time series tasks.

        Notes:
        - Does not support leaveout mode;
        - Kfold mode based on backtesting results from PyCaret's create_model / tune_model steps;
        - In holdout/custom/train-only modes:
          * Training side uniformly uses average metrics from backtesting results as train;
          * Testing side prioritizes scores(test_data=...) or self.test_data from initialization;
            If neither exists, falls back to internal y_test (if available).
        """
        allowed_modes = {"auto", "holdout", "kfold", "custom", "train-only", "leaveout"}
        if mode not in allowed_modes:
            raise ValueError(f"Time series task does not support evaluation mode '{mode}', available options: {sorted(allowed_modes)}")

        if mode == "leaveout":
            raise ValueError("Time series tasks currently do not support leaveout mode")

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
            result = self._scores_time_series_kfold(metrics=metrics)
        elif resolved_mode == "holdout":
            result = self._scores_time_series_holdout(
                metrics=metrics,
                external_test_data=external_test_data,
            )
        elif resolved_mode == "custom":
            result = self._scores_time_series_custom(
                metrics=metrics,
                external_test_data=external_test_data,
            )
        elif resolved_mode == "train-only":
            result = self._scores_time_series_train_only(
                metrics=metrics,
                external_test_data=external_test_data,
            )
        else:
            raise RuntimeError(f"Internal error: unhandled time series evaluation mode '{resolved_mode}'")

        return result

    def _get_time_series_cv_results(self) -> Optional[pd.DataFrame]:
        """
        Try to retrieve cross-validation results table (ts_cv_results) for current time series model.

        Priority:
        1. First find entry in model list where model matches self.current_model and extra contains ts_cv_results;
        2. If not found, fall back to finding most recent entry containing ts_cv_results by searching backwards through all models.
        """
        usecase = getattr(self.experiment, "_ml_usecase", None)
        if usecase is None or "TIME_SERIES" not in str(usecase):
            return None

        # Priority: find record matching current model object
        for info in self.models.values():
            if info.model is self.current_model and isinstance(info.extra, dict):
                cv_df = info.extra.get("ts_cv_results")
                if isinstance(cv_df, pd.DataFrame):
                    return cv_df

        # Fallback: search backwards chronologically to find most recent model with ts_cv_results
        for info in reversed(list(self.models.values())):
            if isinstance(info.extra, dict):
                cv_df = info.extra.get("ts_cv_results")
                if isinstance(cv_df, pd.DataFrame):
                    return cv_df

        return None

    def _get_time_series_train_metrics(self) -> Dict[str, float]:
        """
        Get metrics used to represent "training performance" in time series tasks.

        Priority: use Mean row from ts_cv_results as proxy metric for overall training;
        If backtesting results not found, fall back to current model's record in StoredModel.metrics.
        """
        cv_df = self._get_time_series_cv_results()
        if cv_df is not None:
            tmp = cv_df.copy()
            # Try to use Mean row; if not exists, do column average on non-SD rows
            metrics_series = None
            if any(str(idx).lower() == "mean" for idx in tmp.index):
                metrics_series = tmp.loc[[idx for idx in tmp.index if str(idx).lower() == "mean"][0]]
            else:
                if any(str(idx).lower() == "sd" for idx in tmp.index):
                    tmp = tmp.drop(index=[idx for idx in tmp.index if str(idx).lower() == "sd"][0])
                metrics_series = tmp.mean(axis=0)

            # Remove non-metric columns (e.g. cutoff)
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

    def _get_internal_time_series_test_df(self) -> Optional[pd.DataFrame]:
        """
        Try to get internal y_test from PyCaret configuration and construct a DataFrame containing only the target column.
        """
        try:
            y_test = self.experiment.get_config("y_test")
        except Exception:
            return None

        if y_test is None:
            return None

        target_col = self.target
        if target_col is None:
            # If target not explicitly set, unify column name as 'y'
            target_col = "y"
        return pd.DataFrame({target_col: y_test})

    def _eval_time_series_on_test(
        self,
        test_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Use current model to predict on given test set in time series tasks and manually calculate metrics.

        Requirements:
        - test_df must contain at least one target column:
          * If self.target is not None, try to use this column;
          * If self.target is None and only one column exists, use that column;
        """
        if self.current_model is None:
            raise ValueError("No evaluable time series model currently available")

        target_col = self.target
        if target_col is not None and target_col in test_df.columns:
            y_true = test_df[target_col]
        else:
            # If target column not found and only one column exists, fall back to using that column
            if target_col is None and test_df.shape[1] == 1:
                y_true = test_df.iloc[:, 0]
            else:
                raise ValueError(
                    "Time series test set lacks available target column, cannot calculate evaluation metrics"
                )

        n_periods = len(y_true)
        if n_periods <= 0:
            raise ValueError("Time series test set is empty, cannot calculate evaluation metrics")

        # Delay importing sktime's ForecastingHorizon to avoid adding dependency burden when time series tasks are not used
        try:
            from sktime.forecasting.base import ForecastingHorizon
        except ImportError as exc:
            raise ImportError(
                "Time series scores requires sktime support, please ensure PyCaret with time series dependencies is installed."
            ) from exc

        fh = ForecastingHorizon(np.arange(1, n_periods + 1), is_relative=True)
        model = self.current_model

        # Directly call underlying forecaster's predict to forecast future n_periods periods
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

    def _scores_time_series_kfold(
        self,
        metrics: Union[str, List[str]],
    ) -> pd.DataFrame:
        """
        Time series kfold mode: construct folds and average results based on ts_cv_results.
        """
        cv_df = self._get_time_series_cv_results()
        if cv_df is None:
            raise ValueError(
                "Backtesting results for current time series model not found, cannot calculate scores in kfold mode."
            )

        df = cv_df.copy()

        # Keep only metric columns, remove non-metric information like cutoff
        if "cutoff" in df.columns:
            df = df.drop(columns=["cutoff"])

        # Treat non-Mean/SD rows as folds
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

    def _scores_time_series_holdout(
        self,
        metrics: Union[str, List[str]],
        external_test_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Time series holdout mode:
        - train row: use average metrics from backtesting results;
        - test row: use external test set (if available), otherwise use internal y_test.
        """
        train_metrics = self._get_time_series_train_metrics()

        # Determine test set source
        if external_test_data is not None:
            test_df = external_test_data
        elif self.test_data is not None:
            test_df = self.test_data
        else:
            internal_test_df = self._get_internal_time_series_test_df()
            if internal_test_df is None:
                raise ValueError(
                    "No available test set found in time series holdout mode: "
                    "Neither scores(test_data=...) provided nor y_test configured in experiment."
                )
            test_df = internal_test_df

        test_metrics = self._eval_time_series_on_test(test_df)

        result_df = pd.DataFrame(
            [train_metrics, test_metrics],
            index=["train", "test"],
        )
        return self._filter_metrics_columns(result_df, metrics)

    def _scores_time_series_custom(
        self,
        metrics: Union[str, List[str]],
        external_test_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Time series custom mode:
        - train row: use average metrics from backtesting results;
        - test row: must use scores(test_data=...) or self.test_data from initialization.
        """
        train_metrics = self._get_time_series_train_metrics()

        if external_test_data is not None:
            test_df = external_test_data
        elif self.test_data is not None:
            test_df = self.test_data
        else:
            raise ValueError(
                "Test set must be provided in time series custom mode: "
                "Please pass in scores(test_data=...), or provide test_data when initializing TimeSeriesML."
            )

        test_metrics = self._eval_time_series_on_test(test_df)

        result_df = pd.DataFrame(
            [train_metrics, test_metrics],
            index=["train", "test"],
        )
        return self._filter_metrics_columns(result_df, metrics)

    def _scores_time_series_train_only(
        self,
        metrics: Union[str, List[str]],
        external_test_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Time series train-only mode:
        - Always return training set performance;
        - If external test set or initialized self.test_data exists, additionally return one test row.
        """
        train_metrics = self._get_time_series_train_metrics()

        rows: List[Dict[str, float]] = [train_metrics]
        indices: List[str] = ["train"]

        test_df: Optional[pd.DataFrame] = None
        if external_test_data is not None:
            test_df = external_test_data
        elif self.test_data is not None:
            test_df = self.test_data

        if test_df is not None:
            test_metrics = self._eval_time_series_on_test(test_df)
            rows.append(test_metrics)
            indices.append("test")

        result_df = pd.DataFrame(rows, index=indices)
        return self._filter_metrics_columns(result_df, metrics)

    def _scores_clustering(
        self,
        mode: str,
        metrics: Union[str, List[str]],
        external_test_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Scores implementation for clustering tasks.

        Only supports train-only semantics:
        - Always evaluate internal clustering metrics on training data;
        - If external test set or self.test_data provided during initialization exists, additionally evaluate test set;
        - Only return internal metrics that don't depend on true labels (e.g. Silhouette / CHS / DB),
          external metrics requiring ground truth (e.g. Homogeneity / ARI / Completeness) are not calculated.
        """
        if mode != "train-only":
            raise ValueError("Clustering tasks currently only support train-only mode")

        self._ensure_setup()
        if self.current_model is None:
            raise ValueError("No evaluable clustering model currently available, please create a model first")

        # Get feature columns and preprocessing pipeline to ensure consistency with training phase
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
        train_metrics = self._eval_clustering_on_data(self.data, feature_cols)
        rows.append(train_metrics)
        indices.append("train")

        # Test set (optional): external_test_data first priority, then self.test_data
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
            test_metrics = self._eval_clustering_on_data(test_df, feature_cols)
            rows.append(test_metrics)
            indices.append("test")

        result_df = pd.DataFrame(rows, index=indices)
        return self._filter_metrics_columns(result_df, metrics)

    def _eval_clustering_on_data(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
    ) -> Dict[str, float]:
        """
        Calculate internal clustering metrics on given dataset (do not depend on true labels).
        """
        self._ensure_setup()
        if self.current_model is None:
            raise ValueError("No evaluable clustering model currently available")

        # Select feature columns consistent with training
        X = data[feature_cols]

        # Use preprocessing pipeline consistent with training phase
        try:
            pipeline = self.experiment.get_config("pipeline")
        except Exception as exc:
            raise ValueError("Cannot get clustering preprocessing pipeline from experiment configuration") from exc

        X_trans = pipeline.transform(X)

        # Use current clustering model for prediction
        model = self.current_model
        labels = model.predict(X_trans)

        metrics_containers = getattr(self.experiment, "_all_metrics", {})
        scores: Dict[str, float] = {}

        for key, container in metrics_containers.items():
            # Only keep internal metrics that don't require ground truth
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

    def _setup_experiment(self, **kwargs: Any) -> None:
        """Setup is not implemented in the base class; override in subclass."""
        raise NotImplementedError("Subclass must implement _setup_experiment")
