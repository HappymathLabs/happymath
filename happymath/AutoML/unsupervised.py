"""
Unsupervised learning task wrappers.

Convenience interfaces for clustering and anomaly detection.
"""

from __future__ import annotations

from typing import Any, Optional

from pycaret.anomaly import AnomalyExperiment
from pycaret.clustering import ClusteringExperiment

from .base import AutoMLBase


class ClusteringML(AutoMLBase):
    """AutoML wrapper for clustering tasks."""

    def __init__(
        self,
        data: Any,
        test_data: Optional[Any] = None,
        seed: int = 42,
        n_jobs: int = -1,
        verbose: bool = False,
        html: bool = False,
        primary_metric: Optional[str] = None,
        **setup_kwargs: Any,
    ) -> None:
        self.seed = seed
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.html = html

        metric = primary_metric or "Silhouette"
        super().__init__(
            data=data,
            target=None,
            test_data=test_data,
            primary_metric=metric,
            **setup_kwargs,
        )

    def _setup_experiment(self, **kwargs: Any) -> None:
        setup_params = {
            "data": self.data,
            "session_id": self.seed,
            "n_jobs": self.n_jobs,
            "html": self.html,
            "verbose": self.verbose,
        }
        setup_params.update(kwargs)

        self.experiment = ClusteringExperiment()
        self.experiment.setup(**setup_params)
        self.is_setup = True

    def create(
        self,
        model: str = "kmeans",
        num_clusters: int = 4,
        verbose: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a clustering model and record its metrics."""
        verbose_flag = self.verbose if verbose is None else verbose
        cluster_model = self.experiment.create_model(
            estimator=model,
            num_clusters=num_clusters,
            verbose=verbose_flag,
            **kwargs,
        )

        results = self.experiment.pull()
        self.results = results
        label = self._safe_get_model_name(cluster_model)
        self._store_model_with_metrics(
            cluster_model,
            model_name=f"{model}_clusters_{num_clusters}",
            results_df=results,
            model_label=label,
            additional_info={"num_clusters": num_clusters},
        )
        self.current_model = cluster_model
        return cluster_model

    def assign(self, model: Optional[Any] = None):
        """Assign cluster labels to the data."""
        target_model = model or self.current_model
        if target_model is None:
            raise ValueError("No clustering model available")
        return self.experiment.assign_model(target_model)


class AnomalyML(AutoMLBase):
    """AutoML wrapper for anomaly detection tasks."""

    def __init__(
        self,
        data: Any,
        test_data: Optional[Any] = None,
        fraction: float = 0.05,
        seed: int = 42,
        n_jobs: int = -1,
        verbose: bool = False,
        html: bool = False,
        primary_metric: Optional[str] = None,
        **setup_kwargs: Any,
    ) -> None:
        self.seed = seed
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.html = html
        self.fraction = fraction

        metric = primary_metric or "AUC"
        super().__init__(
            data=data,
            target=None,
            test_data=test_data,
            primary_metric=metric,
            **setup_kwargs,
        )

    def _setup_experiment(self, **kwargs: Any) -> None:
        setup_params = {
            "data": self.data,
            "session_id": self.seed,
            "n_jobs": self.n_jobs,
            "html": self.html,
            "verbose": self.verbose,
        }
        setup_params.update(kwargs)

        self.experiment = AnomalyExperiment()
        self.experiment.setup(**setup_params)
        self.is_setup = True

    def create(
        self,
        model: str = "iforest",
        fraction: Optional[float] = None,
        verbose: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """Create an anomaly detection model."""
        verbose_flag = self.verbose if verbose is None else verbose
        frac = fraction if fraction is not None else self.fraction

        anomaly_model = self.experiment.create_model(
            estimator=model,
            fraction=frac,
            verbose=verbose_flag,
            **kwargs,
        )

        results = self.experiment.pull()
        self.results = results
        label = self._safe_get_model_name(anomaly_model)
        self._store_model_with_metrics(
            anomaly_model,
            model_name=f"{model}_frac_{frac}",
            results_df=results,
            model_label=label,
            additional_info={"fraction": frac},
        )
        self.current_model = anomaly_model
        return anomaly_model

    def assign(self, model: Optional[Any] = None):
        """Label anomalies in the data."""
        target_model = model or self.current_model
        if target_model is None:
            raise ValueError("No anomaly detection model available")
        return self.experiment.assign_model(target_model)
