import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from happymath.AutoML import AnomalyML, ClusteringML, TimeSeriesML


def test_clustering_create_and_assign():
    X, _ = make_blobs(n_samples=60, centers=3, random_state=42)
    df = pd.DataFrame(X, columns=["f1", "f2"])

    cluster = ClusteringML(data=df, seed=42, verbose=False, html=False)
    model = cluster.create(model="kmeans", num_clusters=3, verbose=False)
    assert model is not None

    assigned = cluster.assign()
    assert "Cluster" in assigned.columns

    best_model, metrics = cluster.get_best_model()
    assert best_model is not None
    assert "Silhouette" in metrics


def test_anomaly_create_and_assign():
    X, _ = make_blobs(n_samples=80, centers=3, random_state=0)
    df = pd.DataFrame(X, columns=["x", "y"])

    anomaly = AnomalyML(data=df, fraction=0.1, seed=42, verbose=False, html=False)
    model = anomaly.create(model="iforest", verbose=False)
    assert model is not None

    labeled = anomaly.assign()
    assert "Anomaly" in labeled.columns
    best_model, metrics = anomaly.get_best_model()
    assert best_model is not None
    assert isinstance(metrics, dict)


def test_time_series_compare_and_predict():
    data = pd.Series(np.sin(np.arange(48)), name="value")

    ts = TimeSeriesML(data=data, fh=3, fold=2, seed=42, verbose=False, html=False)
    best = ts.compare(include=["naive", "arima"], sort="MASE", verbose=False)
    assert best is not None

    forecast = ts.predict(fh=3, verbose=False)
    assert "y_pred" in forecast.columns

    best_model, metrics = ts.get_best_model()
    assert best_model is not None
    assert "MASE" in metrics
