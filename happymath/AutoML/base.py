"""
AutoML base classes.

Core base for the HappyMath AutoML framework: data loading, experiment setup,
model storage and evaluation utilities; all task wrappers derive from this.
"""

from __future__ import annotations

import inspect
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

os.environ["PYCARET_CUSTOM_LOGGING_PATH"] = os.devnull
os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "CRITICAL"
os.environ["CATBOOST_ALLOW_WRITING_FILES"] = "0"

import numpy as np
import pandas as pd

from .core.font import ChineseFontMixin
from .core.scores import get_scores_strategy
from .core.utils import (
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


class AutoMLBase(ChineseFontMixin):
    """
    HappyMath AutoML base class.

    Encapsulates the PyCaret experiment lifecycle and provides unified data loading, metric handling, and model management.
    """

    primary_metric: Optional[str] = None

    def __init__(
        self,
        data: DataLike,
        target: TargetLike = None,
        test_data: Optional[DataLike] = None,
        primary_metric: Optional[str] = None,
        **setup_kwargs: Any,
    ) -> None:
        # Disable CatBoost file artifacts
        silence_catboost_artifacts()

        # Data loading and validation
        self.data, normalized_target = DataLoader.load(data, target)
        self.target = normalized_target
        DataLoader.validate(self.data, self.target)

        # Test data processing
        if test_data is not None:
            self.test_data, _ = DataLoader.load(test_data, target=None)
        else:
            self.test_data = None

        # Common attribute initialization
        self.primary_metric = primary_metric or self.primary_metric
        self.setup_kwargs = setup_kwargs
        self.experiment = None
        self._model_storage = ModelStorage()
        self.current_model: Optional[Any] = None
        self.is_setup = False
        self.results = None
        self.verbose = getattr(self, "verbose", False)

        # Auto-execute experiment initialization
        self._setup_experiment(**setup_kwargs)

    # ------------------------------------------------------------------
    # Backward compatible property for models
    # ------------------------------------------------------------------
    @property
    def models(self) -> Dict[str, StoredModel]:
        """Backward compatible: access model dictionary."""
        return self._model_storage.models

    # ------------------------------------------------------------------
    # Metric-related tools (delegate to utility module)
    # ------------------------------------------------------------------
    def _get_metric_direction(self, metric: str) -> str:
        """Determine metric optimization direction from common names."""
        return get_metric_direction(metric)

    def _is_better_score(self, new_score: float, current_best: Optional[float]) -> bool:
        """Determine whether a new score is better based on metric direction."""
        return is_better_score(new_score, current_best, self.primary_metric)

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
        return extract_metrics_from_results(results, model_label)

    def _filter_kwargs_for(self, func: Any, base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter unsupported kwargs by function signature."""
        return filter_kwargs_for(func, base_kwargs)

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
            has_cutoff_col = "cutoff" in results_df.columns
            has_mean_row = any(str(idx).lower() == "mean" for idx in results_df.index)
            if has_cutoff_col and has_mean_row:
                extra.setdefault("ts_cv_results", results_df.copy())

        self._model_storage.store(model, model_name, metrics, extra)

    def get_best_model(self) -> Tuple[Any, Dict[str, Any]]:
        """Return the best model based on primary_metric."""
        model, metrics = self._model_storage.get_best(
            self.primary_metric,
            lambda new, old: self._is_better_score(new, old)
        )
        self.current_model = model
        return model, metrics

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
        from typing import Any as _Any

        self._ensure_setup()
        estimator = estimator or self.current_model
        if estimator is None:
            raise ValueError("No model available for plotting")

        # Special handling for decision tree visualization: use custom implementation to avoid relying on PyCaret's internal tree plotting logic
        if plot_type in {"tree", "tree_text"}:
            return self._plot_decision_tree_auto(
                estimator=estimator,
                plot_type=plot_type,
                scale=scale,
                save=save,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                legend_title=legend_title,
                legend_labels=legend_labels,
                figsize=figsize,
                plot_kwargs=plot_kwargs or {},
                font_sizes=font_sizes,
                verbose=verbose,
            )

        verbose_flag = self.verbose if verbose is None else verbose

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

    def _plot_decision_tree_auto(
        self,
        estimator: Any,
        plot_type: str,
        scale: float,
        save: bool,
        title: Optional[str],
        xlabel: Optional[str],
        ylabel: Optional[str],
        legend_title: Optional[str],
        legend_labels: Optional[List[str]],
        figsize: Tuple[int, int],
        plot_kwargs: Dict[str, Any],
        font_sizes: Optional[Dict[str, Union[int, float]]],
        verbose: Optional[bool],
    ) -> Optional[str]:
        """
        Decision tree visualization (graph and text), only supports single decision tree models.

        - plot_type='tree'     Use graphviz to draw the structure diagram;
        - plot_type='tree_text' Use sklearn.tree.export_text to output text structure.
        """
        from sklearn.tree import BaseDecisionTree, export_graphviz, export_text
        from sklearn.pipeline import Pipeline

        # Only supports single decision tree models, unpack Pipeline if necessary
        base_estimator = estimator
        if isinstance(estimator, Pipeline):
            if not estimator.steps:
                raise TypeError("Pipeline has no estimator, cannot perform decision tree visualization.")
            base_estimator = estimator.steps[-1][1]

        if not isinstance(base_estimator, BaseDecisionTree):
            raise TypeError(
                "plot_type='tree' and 'tree_text' currently only support single decision tree models "
                "(e.g., DecisionTreeClassifier / DecisionTreeRegressor)."
            )

        # Get preprocessed feature names to ensure consistency with user-provided data column names (including Chinese)
        X_train_transformed = self.get_config("X_train_transformed")
        if X_train_transformed is None or not hasattr(X_train_transformed, "columns"):
            raise ValueError("Unable to get X_train_transformed from config for decision tree visualization.")
        feature_names = list(X_train_transformed.columns)

        # Class names for classification tasks (if available)
        class_names: Optional[List[str]] = None
        if hasattr(base_estimator, "classes_"):
            class_names = [str(c) for c in base_estimator.classes_]

        # Text format decision tree
        if plot_type == "tree_text":
            # Use sklearn's export function to ensure reliable structure and compatibility with Chinese feature names
            tree_text = export_text(
                base_estimator,
                feature_names=feature_names,
            )

            plot_name = "Decision_Tree_Text"
            base_plot_filename = f"{plot_name}.txt"

            if save:
                import os

                if isinstance(save, str):
                    plot_filename = os.path.join(save, base_plot_filename)
                else:
                    plot_filename = base_plot_filename
                with open(plot_filename, "w", encoding="utf-8") as f:
                    f.write(tree_text)
                return plot_filename
            else:
                # Return text content directly when not saving, for caller to process further
                return tree_text

        # Graphical decision tree: use graphviz for plotting
        if plot_type == "tree":
            try:
                import graphviz
            except ImportError as exc:
                raise ImportError(
                    "Using plot_type='tree' requires graphviz library. Please run: pip install graphviz, "
                    "and ensure Graphviz executable (e.g., dot) is installed on your system."
                ) from exc

            # Export DOT format string
            dot_data = export_graphviz(
                base_estimator,
                out_file=None,
                feature_names=feature_names,
                class_names=class_names,
                filled=True,
                rounded=True,
                special_characters=True,
                proportion=False,
                precision=2,
            )

            # Set Chinese font for nodes and edges to ensure Chinese feature names display correctly
            chinese_font = self._get_chinese_font()
            lines = dot_data.splitlines()
            if lines:
                insert_idx = 1 if len(lines) > 1 else 0
                lines.insert(insert_idx, f'node [fontname="{chinese_font}"];')
                lines.insert(insert_idx + 1, f'edge [fontname="{chinese_font}"];')
                dot_data = "\n".join(lines)

            # Generate graphviz Source object
            src = graphviz.Source(dot_data, format="png")
            # Try to improve image resolution: fixed high dpi (equivalent to scale=5)
            try:
                dpi_val = int(300 * 5)
                src.graph_attr.update(dpi=str(dpi_val))
            except Exception:
                # Ignore if dpi attribute not supported in some environments
                pass

            plot_name = "Decision_Tree"
            base_plot_filename = f"{plot_name}.png"

            if save:
                import os

                # filename is base name without extension, graphviz will add .png automatically
                if isinstance(save, str):
                    filename_base = os.path.join(save, plot_name)
                else:
                    filename_base = plot_name
                # render returns the path of the generated file (with extension)
                output_path = src.render(filename=filename_base, cleanup=True)
                return output_path
            else:
                # Return Source directly in notebook for self-rendering
                return src  # type: ignore[return-value]

        # Theoretically should not reach here, defensive return
        raise ValueError(f"Unsupported decision tree plot type: {plot_type!r}")

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
        return self._model_storage.get_all_names()

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

        # Parse task type
        task_type = self._get_task_type()

        # Parse external test set (if any)
        external_test_df: Optional[pd.DataFrame] = None
        if test_data is not None:
            target_for_loading: Optional[TargetLike] = self.target if task_type != "clustering" else None
            external_test_df, _ = DataLoader.load(test_data, target=target_for_loading)

        # Parse train_size / fold with clear priority and internal default values
        effective_train_size = self._get_effective_train_size(train_size)
        effective_fold = self._get_effective_fold(fold)

        mode_normalized = (mode or "auto").lower()

        # Get evaluation strategy and execute
        strategy = get_scores_strategy(self, task_type)
        return strategy.evaluate(
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

    def _get_task_type(self) -> str:
        """Get task type as internal unified label."""
        usecase = getattr(self.experiment, "_ml_usecase", None)
        usecase_str = str(usecase) if usecase is not None else ""
        if "TIME_SERIES" in usecase_str:
            return "time_series"
        elif "CLUSTERING" in usecase_str:
            return "clustering"
        elif "CLASSIFICATION" in usecase_str or "REGRESSION" in usecase_str:
            return "supervised"
        else:
            raise NotImplementedError("Current AutoML task type does not support scores() interface")

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

    def _setup_experiment(self, **kwargs: Any) -> None:
        """Setup is not implemented in the base class; override in subclass."""
        raise NotImplementedError("Subclass must implement _setup_experiment")
