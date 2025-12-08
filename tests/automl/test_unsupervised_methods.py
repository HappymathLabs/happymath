import numpy as np
import pandas as pd
import pytest

from happymath.AutoML import AnomalyML, ClusteringML


@pytest.fixture(scope="module")
def cluster_df():
    rng = np.random.default_rng(42)
    centers = np.array([[0, 0], [5, 5], [-5, 5]])
    samples = []
    for center in centers:
        samples.append(center + rng.normal(scale=0.5, size=(60, 2)))
    data = np.vstack(samples)
    return pd.DataFrame(data, columns=["x1", "x2"])


@pytest.fixture(scope="module")
def anomaly_df():
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, size=(120, 2))
    anomalies = rng.normal(5, 0.5, size=(10, 2))
    data = np.vstack([normal, anomalies])
    return pd.DataFrame(data, columns=["x", "y"])


def test_clustering_workflow(cluster_df, tmp_path, monkeypatch):
    clu = ClusteringML(
        data=cluster_df,
        seed=42,
        verbose=False,
        html=False,
    )

    km = clu.create(model="kmeans", num_clusters=3, verbose=False)
    assert km is not None

    assigned = clu.assign()
    assert "Cluster" in assigned.columns

    metrics = clu.get_metrics()
    assert isinstance(metrics, pd.DataFrame)

    results = clu.get_results()
    assert not results.empty

    leaderboard = clu.get_leaderboard()
    assert isinstance(leaderboard, pd.DataFrame)
    assert not leaderboard.empty

    best_model, best_metrics = clu.get_best_model()
    assert isinstance(best_metrics, dict)

    captured = {}

    def fake_plot_model(**kwargs):
        captured["kwargs"] = kwargs
        return "cluster-plot"

    monkeypatch.setattr(clu.experiment, "plot_model", fake_plot_model)
    result = clu.plot(plot_type="cluster", title="Clustering Plot", verbose=False)
    assert result == "cluster-plot"
    assert "kwargs" in captured

    model_path = tmp_path / "cluster_model"
    clu.save(str(model_path))
    assert clu.load(str(model_path)) is not None


def test_anomaly_workflow(anomaly_df, tmp_path, monkeypatch):
    ano = AnomalyML(
        data=anomaly_df,
        fraction=0.1,
        seed=42,
        verbose=False,
        html=False,
    )

    iforest = ano.create(model="iforest", verbose=False)
    assert iforest is not None

    labeled = ano.assign()
    assert {"Anomaly", "Anomaly_Score"}.issubset(labeled.columns)

    metrics = ano.get_metrics()
    assert isinstance(metrics, pd.DataFrame)

    results = ano.get_results()
    assert not results.empty

    leaderboard = ano.get_leaderboard()
    assert isinstance(leaderboard, pd.DataFrame)

    best_model, best_metrics = ano.get_best_model()
    assert isinstance(best_metrics, dict)

    captured = {}

    def fake_plot_model(**kwargs):
        captured["kwargs"] = kwargs
        return "anomaly-plot"

    monkeypatch.setattr(ano.experiment, "plot_model", fake_plot_model)
    result = ano.plot(plot_type="tsne", verbose=False)
    assert result == "anomaly-plot"
    assert "kwargs" in captured

    model_path = tmp_path / "anomaly_model"
    ano.save(str(model_path))
    assert ano.load(str(model_path)) is not None
