"""
Supervised learning task wrappers.

Beginner-friendly interfaces for classification and regression by extending
AutoMLBase to manage the full PyCaret experiment lifecycle.
"""

from __future__ import annotations

from typing import Any, Optional

from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment

from .base import AutoMLBase


class ClassificationML(AutoMLBase):
    """AutoML wrapper for classification tasks."""

    def __init__(
        self,
        data: Any,
        target: str,
        test_data: Optional[Any] = None,
        train_size: float = 0.7,
        fold: int = 5,
        seed: int = 42,
        n_jobs: int = -1,
        verbose: bool = False,
        html: bool = False,
        primary_metric: Optional[str] = None,
        **setup_kwargs: Any,
    ) -> None:
        self.train_size = train_size
        self.fold = fold
        self.seed = seed
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.html = html

        metric = primary_metric or "Accuracy"
        super().__init__(
            data=data,
            target=target,
            test_data=test_data,
            primary_metric=metric,
            **setup_kwargs,
        )

    def _setup_experiment(self, **kwargs: Any) -> None:
        setup_params = {
            "data": self.data,
            "target": self.target,
            "train_size": self.train_size,
            "test_data": self.test_data,
            "fold": self.fold,
            "session_id": self.seed,
            "n_jobs": self.n_jobs,
            "html": self.html,
            "verbose": self.verbose,
        }
        setup_params.update(kwargs)

        self.experiment = ClassificationExperiment()
        self.experiment.setup(**setup_params)
        self.is_setup = True


class RegressionML(AutoMLBase):
    """AutoML wrapper for regression tasks."""

    def __init__(
        self,
        data: Any,
        target: str,
        test_data: Optional[Any] = None,
        train_size: float = 0.7,
        fold: int = 5,
        seed: int = 42,
        n_jobs: int = -1,
        verbose: bool = False,
        html: bool = False,
        primary_metric: Optional[str] = None,
        **setup_kwargs: Any,
    ) -> None:
        self.train_size = train_size
        self.fold = fold
        self.seed = seed
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.html = html

        metric = primary_metric or "MAE"
        super().__init__(
            data=data,
            target=target,
            test_data=test_data,
            primary_metric=metric,
            **setup_kwargs,
        )

    def _setup_experiment(self, **kwargs: Any) -> None:
        setup_params = {
            "data": self.data,
            "target": self.target,
            "train_size": self.train_size,
            "test_data": self.test_data,
            "fold": self.fold,
            "session_id": self.seed,
            "n_jobs": self.n_jobs,
            "html": self.html,
            "verbose": self.verbose,
        }
        setup_params.update(kwargs)

        self.experiment = RegressionExperiment()
        self.experiment.setup(**setup_params)
        self.is_setup = True

    def plot(
        self,
        estimator: Optional[Any] = None,
        plot: str = "residuals",
        scale: float = 1.0,
        save: bool = False,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        legend_title: Optional[str] = None,
        figsize: tuple[int, int] = (10, 6),
        plot_kwargs: Optional[dict] = None,
        verbose: Optional[bool] = None,
    ):
        """Issue a gentle hint when plot type is not typical for regression."""
        regression_plots = {"residuals", "error", "cooks", "feature", "learning"}
        if plot not in regression_plots:
            print(f"Warning: plot type '{plot}' may not be suited for regression (可能不适用于回归任务)")
        return super().plot(
            estimator=estimator,
            plot=plot,
            scale=scale,
            save=save,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            legend_title=legend_title,
            figsize=figsize,
            plot_kwargs=plot_kwargs,
            verbose=verbose,
        )
