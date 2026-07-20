import pandas as pd
from sklearn.datasets import load_diabetes, load_iris

from happymath.AutoML import ClassificationML, RegressionML


def _prepare_classification_data():
    """Construct Iris dataset for classification testing."""
    iris = load_iris(as_frame=True)
    data = iris.data.copy()
    data["target"] = iris.target
    return data


def _prepare_regression_data():
    """Construct Diabetes dataset for regression testing."""
    diabetes = load_diabetes(as_frame=True)
    data = diabetes.data.copy()
    data["target"] = diabetes.target
    return data


def test_classification_create_tune_blend_predict():
    data = _prepare_classification_data()
    clf = ClassificationML(
        data=data,
        target="target",
        train_size=0.8,
        fold=2,
        seed=42,
        verbose=False,
        html=False,
    )

    lr_model = clf.create("lr", verbose=False)
    dt_model = clf.create("dt", verbose=False)
    assert lr_model is not None and dt_model is not None

    tuned_model = clf.tune(estimator=lr_model, n_iter=2, verbose=False)
    assert tuned_model is not None

    blended_model = clf.blend(method="soft", verbose=False)
    assert blended_model is not None

    predictions = clf.predict(data=data.head())
    assert isinstance(predictions, pd.DataFrame)
    assert any(col in predictions.columns for col in ["Label", "prediction_label"])

    best_model, metrics = clf.get_best_model()
    assert best_model is not None
    assert "Accuracy" in metrics


def test_multiclass_auc_uses_probabilities():
    data = _prepare_classification_data()
    clf = ClassificationML(
        data=data,
        target="target",
        train_size=0.8,
        fold=2,
        seed=42,
        verbose=False,
        html=False,
        n_jobs=1,
    )

    clf.create("lr", verbose=False)
    auc = float(clf.get_results().loc["Mean", "AUC"])
    assert auc > 0.99


def test_regression_compare_ensemble_predict():
    data = _prepare_regression_data()
    reg = RegressionML(
        data=data,
        target="target",
        train_size=0.8,
        fold=2,
        seed=42,
        verbose=False,
        html=False,
    )

    best_model = reg.compare(include=["lr", "dt"], sort="MAE", verbose=False)
    assert best_model is not None

    ensemble_model = reg.ensemble(method="Bagging", n_estimators=5, verbose=False)
    assert ensemble_model is not None

    predictions = reg.predict(data=data.head())
    assert isinstance(predictions, pd.DataFrame)
    assert any(col in predictions.columns for col in ["Label", "prediction_label"])

    best_model, metrics = reg.get_best_model()
    assert "MAE" in metrics
