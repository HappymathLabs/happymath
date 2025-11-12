"""
Time series task wrapper.
"""

from __future__ import annotations

from typing import Any, Optional

from pycaret.time_series import TSForecastingExperiment

from .base import AutoMLBase


class TimeSeriesML(AutoMLBase):
    """AutoML wrapper for time series forecasting tasks."""

    def __init__(
        self,
        data: Any,
        target: Optional[str] = None,
        test_data: Optional[Any] = None,
        fh: int = 12,
        fold: int = 3,
        seasonal_period: Optional[int] = None,
        seed: int = 42,
        n_jobs: int = -1,
        verbose: bool = False,
        html: bool = False,
        primary_metric: Optional[str] = None,
        **setup_kwargs: Any,
    ) -> None:
        self.fh = fh
        self.fold = fold
        self.seasonal_period = seasonal_period
        self.seed = seed
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.html = html

        metric = primary_metric or "MASE"
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
            "fh": self.fh,
            "fold": self.fold,
            "session_id": self.seed,
            "seasonal_period": self.seasonal_period,
            "n_jobs": self.n_jobs,
            "html": self.html,
            "verbose": self.verbose,
        }
        setup_params.update(kwargs)

        self.experiment = TSForecastingExperiment()
        self.experiment.setup(**setup_params)
        self.is_setup = True

    def predict(
        self,
        estimator: Optional[Any] = None,
        fh: Optional[int] = None,
        X: Optional[Any] = None,
        return_pred_int: bool = False,
        verbose: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Forecast with optional horizon and parameters."""
        self._ensure_setup()
        predictor = estimator or self.current_model
        if predictor is None:
            raise ValueError("No time-series model available for prediction")

        verbose_flag = self.verbose if verbose is None else verbose
        predict_fn = self.experiment.predict_model
        return predict_fn(
            estimator=predictor,
            fh=fh,
            X=X,
            return_pred_int=return_pred_int,
            verbose=verbose_flag,
            **kwargs,
        )
