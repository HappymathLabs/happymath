# HappyMath AutoML 案例演示

本页通过一个完整脚本展示 `happymath.AutoML` 的主要接口，包括分类、回归、聚类和异常检测任务。所有代码已在 `happymath` conda 环境下通过 `conda run -n happymath python automl_examples.py` 验证运行成功。

> 提示：为控制运行时间，以下示例对 `fold`、`n_iter`、`n_estimators` 等参数做了适当降低。生产环境可按需调高。

## 完整脚本

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from happymath.AutoML import ClassificationML, RegressionML, ClusteringML, AnomalyML


def main():
    # ================================================================
    # 1. 分类任务：Iris
    # ================================================================
    print("Classification: Iris")
    iris = load_iris(as_frame=True)
    data = iris.data.copy()
    data["target"] = iris.target

    clf = ClassificationML(
        data=data,
        target="target",
        train_size=0.8,
        fold=2,
        seed=42,
        verbose=False,
        html=False,
    )

    # 1.1 模型对比（仅对比 lr 与 dt，避免 LightGBM 等依赖影响）
    best = clf.compare(include=["lr", "dt"], sort="Accuracy", verbose=False)
    print("  compare best:", best.__class__.__name__)

    # 1.2 创建指定模型
    lr = clf.create("lr", verbose=False)
    dt = clf.create("dt", verbose=False)
    print("  created:", lr.__class__.__name__, dt.__class__.__name__)

    # 1.3 超参数调优
    tuned = clf.tune(estimator=lr, n_iter=2, verbose=False)
    print("  tuned:", tuned.__class__.__name__)

    # 1.4 模型融合（soft voting）
    blended = clf.blend(method="soft", verbose=False)
    print("  blended:", blended.__class__.__name__)

    # 1.5 预测
    predictions = clf.predict(data=data.head())
    print("  predict columns:", predictions.columns.tolist())

    # 1.6 获取最优模型
    best_model, metrics = clf.get_best_model()
    print("  best model:", best_model.__class__.__name__)
    print("  metrics keys:", list(metrics.keys()))

    # ================================================================
    # 2. 回归任务：Diabetes
    # ================================================================
    print("\nRegression: Diabetes")
    diabetes = load_diabetes(as_frame=True)
    reg_data = diabetes.data.copy()
    reg_data["target"] = diabetes.target

    reg = RegressionML(
        data=reg_data,
        target="target",
        train_size=0.8,
        fold=2,
        seed=42,
        verbose=False,
        html=False,
    )

    # 2.1 模型对比
    reg.compare(include=["lr", "dt"], sort="MAE", verbose=False)

    # 2.2 Bagging 集成
    ensemble = reg.ensemble(method="Bagging", n_estimators=5, verbose=False)
    print("  ensemble:", ensemble.__class__.__name__)

    # 2.3 预测
    reg_pred = reg.predict(data=reg_data.head())
    print("  predict columns:", reg_pred.columns.tolist())

    # ================================================================
    # 3. 聚类任务：合成数据
    # ================================================================
    print("\nClustering: synthetic 2D data")
    rng = np.random.default_rng(42)
    centers = np.array([[0, 0], [5, 5], [-5, 5]])
    samples = [center + rng.normal(scale=0.5, size=(60, 2)) for center in centers]
    cluster_df = pd.DataFrame(np.vstack(samples), columns=["x1", "x2"])

    clu = ClusteringML(data=cluster_df, seed=42, verbose=False, html=False)
    clu.create(model="kmeans", num_clusters=3, verbose=False)
    assigned = clu.assign()
    print("  assigned columns:", assigned.columns.tolist())
    print("  cluster counts:\n", assigned["Cluster"].value_counts())

    # ================================================================
    # 4. 异常检测任务：合成数据
    # ================================================================
    print("\nAnomaly: synthetic 2D data")
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, size=(120, 2))
    anomalies = rng.normal(5, 0.5, size=(10, 2))
    anomaly_df = pd.DataFrame(np.vstack([normal, anomalies]), columns=["x", "y"])

    ano = AnomalyML(
        data=anomaly_df,
        fraction=0.1,
        seed=42,
        verbose=False,
        html=False,
    )
    ano.create(model="iforest", verbose=False)
    labeled = ano.assign()
    print("  labeled columns:", labeled.columns.tolist())
    print("  anomaly counts:\n", labeled["Anomaly"].value_counts())

    print("\nAll examples completed.")


if __name__ == "__main__":
    main()
```

## 运行结果示例

```text
Classification: Iris
  compare best: LogisticRegression
  created: LogisticRegression DecisionTreeClassifier
  tuned: LogisticRegression
  blended: LogisticRegression
  predict columns: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                    'petal width (cm)', 'target', 'prediction_label', 'prediction_score']
  best model: LogisticRegression
  metrics keys: ['Model', 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', ...]

Regression: Diabetes
  ensemble: BaggingRegressor
  predict columns: ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6',
                    'target', 'prediction_label']

Clustering: synthetic 2D data
  assigned columns: ['x1', 'x2', 'Cluster']
  cluster counts:
   Cluster
  Cluster 0    60
  Cluster 1    60
  Cluster 2    60

Anomaly: synthetic 2D data
  labeled columns: ['x', 'y', 'Anomaly', 'Anomaly_Score']
  anomaly counts:
   Anomaly
  0    119
  1     11

All examples completed.
```

## 说明

- `compare(include=["lr", "dt"], ...)` 中的 `"lr"` 和 `"dt"` 分别是 PyCaret 中逻辑回归（Logistic Regression）与决策树（Decision Tree）的模型 ID。通过 `include` 限定模型范围可以显著缩短运行时间并避免某些可选依赖缺失导致的问题。
- `blend(method="soft", ...)` 会自动使用当前已保存的多个模型进行软投票融合；在分类任务中要求模型能够输出概率。
- `ensemble(method="Bagging", n_estimators=5, ...)` 对当前模型做 Bagging 集成，`method="Boosting"` 可尝试 AdaBoost 风格的集成（具体可用选项取决于 PyCaret 版本）。
- 聚类与异常检测任务无 `target` 参数，使用 `assign()` 可将簇标签或异常标记写回原始数据。

## 扩展建议

- 在分类 / 回归任务中，训练完成后可调用 `clf.scores(mode="kfold", fold=5)` 查看交叉验证结果，或 `clf.save("my_model")` 保存最优模型。
- 需要中文图表时，可尝试 `clf.plot(plot_type="auc", title="AUC 曲线", xlabel="FPR", ylabel="TPR")`，但部分 plot 类型依赖 PyCaret 与系统字体配置，可能需要在绘图前确保中文字体可用。
