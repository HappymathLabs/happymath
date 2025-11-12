import numpy as np
import pandas as pd
import pytest

from happymath.AutoML.base import AutoMLBase


class DummyExperiment:
    """用于模拟 PyCaret Experiment 的最小实现。"""

    def __init__(self):
        self._frame = pd.DataFrame([{"Accuracy": 0.5}])

    def set_metrics(self, metrics):
        self._frame = pd.DataFrame([metrics])

    def pull(self):
        return self._frame.copy()

    def get_metrics(self):
        return ["Accuracy", "F1", "MAE"]

    def get_leaderboard(self):
        return pd.DataFrame({"Model": ["dummy"], "Accuracy": [0.5]})

    def save_model(self, model, name):
        pass

    def load_model(self, name):
        return object()

    def get_config(self, key=None):
        return {"key": key}


class DummyAutoML(AutoMLBase):
    """用于测试 AutoMLBase 的简单子类。"""

    def _setup_experiment(self, **kwargs):
        self.experiment = DummyExperiment()
        self.is_setup = True


def test_dataframe_loading_with_target_name():
    data = pd.DataFrame({"f1": [1, 2], "f2": [3, 4], "y": [0, 1]})
    automl = DummyAutoML(data=data, target="y", primary_metric="Accuracy")
    assert automl.target == "y"
    pd.testing.assert_frame_equal(automl.data, data)


def test_numpy_loading_with_target_index():
    array = np.array([[1, 2, 0], [3, 4, 1]])
    automl = DummyAutoML(data=array, target=2, primary_metric="Accuracy")
    assert automl.target == "target"
    assert "target" in automl.data.columns
    assert automl.data.shape == (2, 3)


def test_metric_direction_affects_best_model_selection():
    data = pd.DataFrame({"f1": [1, 2], "y": [0, 1]})
    automl = DummyAutoML(data=data, target="y", primary_metric="Accuracy")

    automl.experiment.set_metrics({"Accuracy": 0.7})
    automl._store_model_with_metrics(model="m1", model_name="model_accuracy_1")

    automl.experiment.set_metrics({"Accuracy": 0.9})
    automl._store_model_with_metrics(model="m2", model_name="model_accuracy_2")

    _, metrics = automl.get_best_model()
    assert metrics["Accuracy"] == 0.9


def test_lower_better_metric_comparison():
    data = pd.DataFrame({"f1": [1, 2], "y": [0, 1]})
    automl = DummyAutoML(data=data, target="y", primary_metric="MAE")

    automl.experiment.set_metrics({"MAE": 0.5})
    automl._store_model_with_metrics(model="m1", model_name="model_mae_1")

    automl.experiment.set_metrics({"MAE": 0.3})
    automl._store_model_with_metrics(model="m2", model_name="model_mae_2")

    _, metrics = automl.get_best_model()
    assert metrics["MAE"] == 0.3


def test_get_results_with_cached_value():
    data = pd.DataFrame({"f1": [1, 2], "y": [0, 1]})
    automl = DummyAutoML(data=data, target="y", primary_metric="Accuracy")
    cached = pd.DataFrame({"Accuracy": [0.8]})
    automl.results = cached
    result = automl.get_results()
    pd.testing.assert_frame_equal(result, cached)


def test_get_results_without_cached_value():
    data = pd.DataFrame({"f1": [1, 2], "y": [0, 1]})
    automl = DummyAutoML(data=data, target="y", primary_metric="Accuracy")
    result = automl.get_results()
    assert isinstance(result, pd.DataFrame)


def test_save_without_model_raise_error():
    data = pd.DataFrame({"f1": [1, 2], "y": [0, 1]})
    automl = DummyAutoML(data=data, target="y", primary_metric="Accuracy")
    automl.current_model = None
    with pytest.raises(ValueError):
        automl.save("demo_model")
