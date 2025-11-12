"""
监督学习任务封装。

该模块提供分类与回归两类初学者友好的接口，通过继承 AutoMLBase
来统一管理 PyCaret 实验生命周期。
"""

from __future__ import annotations

from typing import Any, Optional

from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment

from .base import AutoMLBase


class ClassificationML(AutoMLBase):
    """面向分类任务的 AutoML 封装。"""

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
    """面向回归任务的 AutoML 封装。"""

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
        """在回归情境下增加图表适用性的提示。"""
        regression_plots = {"residuals", "error", "cooks", "feature", "learning"}
        if plot not in regression_plots:
            print(f"警告: 图表类型 '{plot}' 可能不适用于回归任务")
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
