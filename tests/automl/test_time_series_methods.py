import numpy as np
import pandas as pd
import pytest

from happymath.AutoML import TimeSeriesML


@pytest.fixture(scope="module")
def ts_series():
    rng = np.random.default_rng(42)
    periods = 40
    index = pd.date_range("2020-01-01", periods=periods, freq="MS")
    values = np.sin(np.arange(periods) / 4) + rng.normal(0, 0.1, periods)
    return pd.Series(values, index=index, name="value")


def test_time_series_full_workflow(ts_series, tmp_path, monkeypatch):
    ts = TimeSeriesML(
        data=ts_series,
        fh=4,
        fold=2,
        seed=42,
        verbose=False,
        html=False,
    )

    best = ts.compare(include=["naive", "snaive"], sort="MASE", verbose=False)
    assert best is not None

    arima = ts.create("arima", verbose=False)
    assert arima is not None

    tuned = ts.tune(estimator=arima, n_iter=1, optimize="MASE", verbose=False)
    assert tuned is not None

    blended = ts.blend(method="mean", verbose=False)
    assert blended is not None

    predictions = ts.predict(fh=5, return_pred_int=True, verbose=False)
    assert isinstance(predictions, pd.DataFrame)
    assert "y_pred" in predictions.columns
    assert len(predictions) == 5

    metrics_df = ts.get_metrics()
    assert "Display Name" in metrics_df.columns

    results_df = ts.get_results()
    assert not results_df.empty

    leaderboard_df = ts.get_leaderboard()
    assert "Model" in leaderboard_df.columns

    best_model, best_metrics = ts.get_best_model()
    assert isinstance(best_metrics, dict)

    eval_called = {"flag": False}

    def fake_evaluate(model):
        eval_called["flag"] = True

    monkeypatch.setattr(ts.experiment, "evaluate_model", fake_evaluate)
    ts.evaluate()
    assert eval_called["flag"] is True

    captured = {}

    def fake_plot_model(
        estimator=None,
        plot=None,
        return_fig=False,
        return_data=False,
        verbose=False,
        display_format=None,
        data_kwargs=None,
        fig_kwargs=None,
        save=False,
    ):
        captured["kwargs"] = {
            "estimator": estimator,
            "plot": plot,
            "fig_kwargs": fig_kwargs,
            "data_kwargs": data_kwargs,
        }
        return "plot-ok"

    monkeypatch.setattr(ts.experiment, "plot_model", fake_plot_model)
    plot_result = ts.plot(
        plot="ts",
        title="时间序列趋势",
        xlabel="日期",
        ylabel="值",
        figsize=(12, 6),
        verbose=False,
    )
    assert plot_result == "plot-ok"
    fig_kwargs = captured["kwargs"]["fig_kwargs"]
    assert fig_kwargs["title"] == "时间序列趋势"
    assert fig_kwargs["figsize"] == (12, 6)

    final_model = ts.finalize()
    assert final_model is not None

    model_path = tmp_path / "ts_model"
    ts.save(str(model_path))
    loaded_model = ts.load(str(model_path))
    assert loaded_model is not None

    config = ts.get_config()
    assert config is not None
