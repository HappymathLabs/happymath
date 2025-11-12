import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris

from happymath.AutoML.supervised import ClassificationML, RegressionML


@pytest.fixture(scope="module")
def iris_df():
    data = load_iris(as_frame=True)
    df = data.frame.copy()
    df["target"] = data.target
    return df


@pytest.fixture(scope="module")
def diabetes_df():
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    df["target"] = data.target
    return df


def test_classification_all_methods(tmp_path, iris_df, monkeypatch):
    clf = ClassificationML(
        data=iris_df,
        target="target",
        train_size=0.8,
        fold=2,
        seed=42,
        verbose=False,
        html=False,
    )

    best = clf.compare(include=["lr", "dt"], sort="Accuracy", verbose=False)
    assert best is not None

    created = clf.create("nb", verbose=False)
    assert created is not None

    tuned = clf.tune(estimator=created, n_iter=2, verbose=False)
    assert tuned is not None

    ensemble_model = clf.ensemble(
        estimator=tuned, method="Bagging", n_estimators=5, verbose=False
    )
    assert ensemble_model is not None

    blended = clf.blend(method="soft", verbose=False)
    assert blended is not None

    stacked = clf.stack(meta_model_fold=2, verbose=False)
    assert stacked is not None

    assert len(clf.get_models()) >= 1

    metrics_df = clf.get_metrics()
    assert "Accuracy" in metrics_df["Display Name"].values

    results_df = clf.get_results()
    assert not results_df.empty

    leaderboard_df = clf.get_leaderboard()
    assert "Model" in leaderboard_df.columns

    eval_called = {"flag": False}

    def fake_evaluate(model):
        eval_called["flag"] = True

    monkeypatch.setattr(clf.experiment, "evaluate_model", fake_evaluate)
    clf.evaluate()
    assert eval_called["flag"] is True

    predictions = clf.predict(data=iris_df.head())
    assert isinstance(predictions, pd.DataFrame)
    assert any(col in predictions.columns for col in ["Label", "prediction_label"])

    final_model = clf.finalize()
    assert final_model is not None

    save_path = tmp_path / "classification_model"
    clf.save(str(save_path))
    loaded = clf.load(str(save_path))
    assert loaded is not None

    config = clf.get_config()
    assert config is not None

    best_model, best_metrics = clf.get_best_model()
    assert clf.primary_metric in best_metrics


def test_classification_plot_kwargs(monkeypatch, iris_df):
    clf = ClassificationML(
        data=iris_df,
        target="target",
        train_size=0.8,
        fold=2,
        seed=42,
        verbose=False,
        html=False,
    )
    clf.create("lr", verbose=False)

    captured = {}

    def fake_plot_model(*, estimator, plot, save, plot_kwargs, verbose, **kwargs):
        captured["plot"] = plot
        captured["plot_kwargs"] = plot_kwargs
        return "OK"

    monkeypatch.setattr(clf.experiment, "plot_model", fake_plot_model)

    result = clf.plot(
        plot="auc",
        title="测试标题",
        xlabel="横轴",
        ylabel="纵轴",
        legend_title="图例",
        figsize=(12, 8),
    )

    assert result == "OK"
    assert captured["plot"] == "auc"
    assert captured["plot_kwargs"]["title"] == "测试标题"
    assert captured["plot_kwargs"]["xlabel"] == "横轴"
    assert captured["plot_kwargs"]["ylabel"] == "纵轴"
    assert captured["plot_kwargs"]["legend_title"] == "图例"
    assert captured["plot_kwargs"]["figsize"] == (12, 8)


def test_classification_best_model_fallback(iris_df, capsys):
    clf = ClassificationML(
        data=iris_df,
        target="target",
        train_size=0.8,
        fold=2,
        seed=42,
        verbose=False,
        html=False,
    )
    clf.create("lr", verbose=False)

    clf.primary_metric = "NonExistingMetric"
    model, metrics = clf.get_best_model()
    captured = capsys.readouterr().out
    assert "NonExistingMetric" in captured
    assert model is not None
    assert isinstance(metrics, dict)


def test_regression_all_methods(tmp_path, diabetes_df, monkeypatch):
    reg = RegressionML(
        data=diabetes_df,
        target="target",
        train_size=0.8,
        fold=2,
        seed=42,
        verbose=False,
        html=False,
    )

    best = reg.compare(include=["lr", "dt"], sort="MAE", verbose=False)
    assert best is not None

    created = reg.create("lr", verbose=False)
    assert created is not None

    tuned = reg.tune(estimator=created, n_iter=2, verbose=False)
    assert tuned is not None

    ensemble_model = reg.ensemble(
        estimator=tuned, method="Bagging", n_estimators=5, verbose=False
    )
    assert ensemble_model is not None

    blended = reg.blend(method="auto", verbose=False)
    assert blended is not None

    stacked = reg.stack(meta_model_fold=2, verbose=False)
    assert stacked is not None

    assert len(reg.get_models()) >= 1

    metrics_df = reg.get_metrics()
    assert "MAE" in metrics_df["Display Name"].values

    results_df = reg.get_results()
    assert not results_df.empty

    leaderboard_df = reg.get_leaderboard()
    assert "Model" in leaderboard_df.columns

    eval_called = {"flag": False}

    def fake_evaluate(model):
        eval_called["flag"] = True

    monkeypatch.setattr(reg.experiment, "evaluate_model", fake_evaluate)
    reg.evaluate()
    assert eval_called["flag"] is True

    predictions = reg.predict(data=diabetes_df.head())
    assert isinstance(predictions, pd.DataFrame)
    assert "Label" in predictions.columns or "prediction_label" in predictions.columns

    final_model = reg.finalize()
    assert final_model is not None

    save_path = tmp_path / "regression_model"
    reg.save(str(save_path))
    loaded = reg.load(str(save_path))
    assert loaded is not None

    config = reg.get_config()
    assert config is not None

    best_model, best_metrics = reg.get_best_model()
    assert reg.primary_metric in best_metrics


def test_regression_plot_warning(monkeypatch, capsys, diabetes_df):
    reg = RegressionML(
        data=diabetes_df,
        target="target",
        train_size=0.8,
        fold=2,
        seed=42,
        verbose=False,
        html=False,
    )
    reg.create("lr", verbose=False)
    monkeypatch.setattr(
        reg.experiment,
        "plot_model",
        lambda **kwargs: "OK",
    )
    reg.plot(plot="auc", verbose=False)
    output = capsys.readouterr().out
    assert "可能不适用于回归任务" in output
