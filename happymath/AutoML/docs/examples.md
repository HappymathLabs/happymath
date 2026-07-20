# HappyMath AutoML 示例

本页示例遵循 PyCaret 官方工作流：先完成实验初始化，再用 `compare_models()` 做模型比较；只有需要超参数优化时才调用 `tune_model()`，只有需要集成学习时才调用 `ensemble_model()`、`blend_models()` 或 `stack_models()`。在 HappyMath 中，实验初始化由 `ClassificationML`、`RegressionML`、`TimeSeriesML` 等任务类自动完成。

所有代码均使用 HappyMath 接口，并应在 conda 的 `happymath` 环境中运行：

```bash
conda run -n happymath python automl_docs_examples.py
```

## 1. 分类：基础模型比较

适用场景：快速选择一个分类模型。`compare()` 不会自动调参，它只比较候选模型在交叉验证中的表现。

```python
from sklearn.datasets import load_iris
from happymath.AutoML import ClassificationML

iris = load_iris(as_frame=True)
clf_data = iris.data.copy()
clf_data["target"] = iris.target

clf = ClassificationML(
    data=clf_data,
    target="target",
    train_size=0.8,
    fold=2,
    seed=42,
    verbose=False,
    html=False,
)

best_clf = clf.compare(include=["lr", "dt"], sort="Accuracy", verbose=False)
print("classification_best", best_clf.__class__.__name__)
print(clf.get_results()[["Model", "Accuracy", "AUC", "F1"]])
```

示例输出：

```text
classification_best LogisticRegression
                       Model  Accuracy     AUC      F1
lr       Logistic Regression    0.9667  0.9981  0.9666
dt  Decision Tree Classifier    0.9500  0.9625  0.9499
```

处理建议：

- 如果任务只要求比较模型效果，到这里即可停止。
- 如果正负类代价不均衡，改用 `sort="F1"`、`sort="Recall"` 或 `sort="AUC"`。
- 如果需要预测概率，用 `predict(raw_score=True)`。

## 2. 分类：比较后再调参

适用场景：已经通过 `compare()` 确定候选模型，需要进一步优化它的超参数。PyCaret 官方示例会先 `create_model()` 或 `compare_models()` 得到模型，再 `tune_model(model)`；HappyMath 中对应为 `create()` / `compare()` 后调用 `tune()`。

```python
tuned_clf = clf.tune(
    estimator=best_clf,
    n_iter=5,
    optimize="Accuracy",
    choose_better=True,
    verbose=False,
    tuner_verbose=False,
)
print("classification_tuned", tuned_clf.__class__.__name__)
print(clf.get_results()[["Accuracy", "AUC", "F1"]])
```

示例输出：

```text
classification_tuned LogisticRegression
      Accuracy  AUC      F1
Fold                     
0       0.9667  0.0  0.9666
1       1.0000  0.0  1.0000
Mean    0.9833  0.0  0.9833
Std     0.0167  0.0  0.0167
```

处理建议：

- `n_iter=5` 适合快速验证；正式实验可增加到 `20`、`50` 或更高。
- `custom_grid={...}` 用于手动指定搜索空间。
- 可通过 `search_library="optuna"` 等参数切换底层搜索库，但需要环境中安装对应依赖。
- `choose_better=True` 适合自动化脚本，因为调参没有提升时会返回输入模型。

## 3. 回归：按误差指标比较并调参

适用场景：回归任务通常关心误差最小化。HappyMath 的 `RegressionML` 默认主指标是 `MAE`，但你也可以改成 `RMSE` 或 `R2`。

```python
from sklearn.datasets import load_diabetes
from happymath.AutoML import RegressionML

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

best_reg = reg.compare(include=["lr", "dt"], sort="MAE", verbose=False)
print("regression_best", best_reg.__class__.__name__)
print(reg.get_results()[["Model", "MAE", "RMSE", "R2"]])

tuned_reg = reg.tune(
    estimator=best_reg,
    n_iter=5,
    optimize="MAE",
    choose_better=True,
    verbose=False,
    tuner_verbose=False,
)
print("regression_tuned", tuned_reg.__class__.__name__)
print(reg.get_results()[["MAE", "RMSE", "R2"]])
```

示例输出：

```text
regression_best LinearRegression
                      Model      MAE     RMSE      R2
lr        Linear Regression  45.6885  56.4420  0.4711
dt  Decision Tree Regressor  66.2042  83.7161 -0.1661
regression_tuned LinearRegression
          MAE     RMSE      R2
Fold                        
0     45.1318  55.5718  0.4717
1     46.2451  57.3122  0.4705
Mean  45.6885  56.4420  0.4711
Std    0.5567   0.8702  0.0006
```

处理建议：

- 误差指标越小越好，例如 `MAE`、`RMSE`、`MAPE`。
- 拟合优度指标越大越好，例如 `R2`。
- `compare(sort=...)` 和 `tune(optimize=...)` 应使用同一建模目标，避免比较和调参方向不一致。

## 4. 需要集成学习时：ensemble、blend、stack

适用场景：题目明确要求 Bagging、Boosting、Voting、Averaging 或 Stacking，或者你正在做模型增强实验。不要把这些步骤作为简单模型比较的默认动作。

```python
lr = clf.create("lr", verbose=False)
dt = clf.create("dt", verbose=False)
nb = clf.create("nb", verbose=False)

bagged_lr = clf.ensemble(
    estimator=lr,
    method="Bagging",
    n_estimators=5,
    optimize="Accuracy",
    verbose=False,
)
print("bagged_model", bagged_lr.__class__.__name__)

blended = clf.blend(
    estimator_list=[lr, dt, nb],
    method="soft",
    optimize="Accuracy",
    verbose=False,
)
print("blended_model", blended.__class__.__name__)

stacked = clf.stack(
    estimator_list=[lr, dt, nb],
    meta_model_fold=2,
    optimize="Accuracy",
    verbose=False,
)
print("stacked_model", stacked.__class__.__name__)
```

示例输出：

```text
bagged_model LogisticRegression
blended_model LogisticRegression
stacked_model LogisticRegression
```

处理建议：

- `ensemble()` 作用于一个基模型，`method="Bagging"` 或 `"Boosting"`。
- `blend()` 需要多个已训练模型，分类中 `soft` 依赖模型概率输出；`auto` 会优先尝试 `soft` 并在不支持时回退。
- `stack()` 需要多个基模型和一个元模型；不传 `meta_model` 时使用 PyCaret 默认策略。
- 融合和堆叠可能提升效果，也可能变差；HappyMath 默认向 PyCaret 传入 `choose_better=True`，因此没有提升时返回值可能是输入模型，而不是 Bagging / Voting / Stacking 类。

## 5. 预测、评估表、最终模型和保存

适用场景：完成模型选择后，需要输出预测结果、报告评估表，或保存最终模型。

```python
pred = clf.predict(estimator=best_clf, data=clf_data.head(), raw_score=True, verbose=False)
print("prediction_columns", pred.columns.tolist())
print(pred.filter(regex="target|prediction").head())

score_table = clf.scores(mode="kfold", metrics=["Accuracy", "F1"], fold=3)
print(score_table)

final_model = clf.finalize(best_clf)
clf.save("automl_classification_final", final_model)
loaded_model = clf.load("automl_classification_final")
print("loaded_model", loaded_model.__class__.__name__)
```

示例输出：

```text
prediction_columns ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'target', 'prediction_label', 'prediction_score_0', 'prediction_score_1', 'prediction_score_2']
   target  prediction_label
0       0                 0
1       0                 0
2       0                 0
3       0                 0
4       0                 0
        Accuracy_train  Accuracy_test  F1_train  F1_test
fold_1          0.9800         1.0000    0.9800   1.0000
fold_2          1.0000         0.9600    1.0000   0.9598
fold_3          0.9800         1.0000    0.9800   1.0000
mean            0.9867         0.9867    0.9867   0.9866
loaded_model Pipeline
```

处理建议：

- `predict(data=...)` 可用于无目标列的新数据；如果不传 `data`，PyCaret 会在可用时使用 holdout/test 数据。
- `scores(mode="custom", test_data=...)` 的监督学习测试集必须包含目标列。
- `finalize()` 只用全量数据重新拟合当前模型，不会重新调参。
- `save()` 保存的是完整 PyCaret pipeline；加载后可继续用于预测。

## 6. 聚类：创建模型并分配簇标签

适用场景：没有目标列，希望把样本分成若干簇。聚类任务通常从 `create(model="kmeans", num_clusters=...)` 开始。

```python
import numpy as np
import pandas as pd
from happymath.AutoML import ClusteringML

rng = np.random.default_rng(42)
cluster_data = pd.DataFrame(
    np.vstack([
        rng.normal([0, 0], 0.5, size=(30, 2)),
        rng.normal([5, 5], 0.5, size=(30, 2)),
        rng.normal([-5, 5], 0.5, size=(30, 2)),
    ]),
    columns=["x1", "x2"],
)

cluster = ClusteringML(cluster_data, seed=42, verbose=False, html=False)
cluster_model = cluster.create(model="kmeans", num_clusters=3, verbose=False)
assigned = cluster.assign(cluster_model)
print("cluster_model", cluster_model.__class__.__name__)
print(assigned["Cluster"].value_counts().sort_index())
print(cluster.scores(mode="train-only", metrics="Silhouette"))
```

示例输出：

```text
cluster_model KMeans
Cluster
Cluster 0    30
Cluster 1    30
Cluster 2    30
Name: count, dtype: int64
       Silhouette
train      0.8888
```

## 7. 异常检测：设置异常比例并打标签

适用场景：没有目标列，希望识别离群样本。`fraction` 是预期异常比例，影响模型标记多少样本为异常。

```python
from happymath.AutoML import AnomalyML

normal = rng.normal(0, 1, size=(80, 2))
outliers = rng.normal(5, 0.5, size=(8, 2))
anomaly_data = pd.DataFrame(np.vstack([normal, outliers]), columns=["x", "y"])

anomaly = AnomalyML(anomaly_data, fraction=0.1, seed=42, verbose=False, html=False)
anomaly_model = anomaly.create(model="iforest", verbose=False)
labeled = anomaly.assign(anomaly_model)
print("anomaly_model", anomaly_model.__class__.__name__)
print(labeled["Anomaly"].value_counts().sort_index())
```

示例输出：

```text
anomaly_model IForest
Anomaly
0    79
1     9
Name: count, dtype: int64
```

处理建议：

- `Anomaly=1` 表示异常样本，`Anomaly_Score` 表示异常程度。
- 如果业务先验认为异常更少，应降低 `fraction`。
- 当前 `scores()` 不支持异常检测任务，通常用 `assign()` 后统计或结合人工标签另行评价。

## 8. 时序预测：比较预测器并生成未来预测

适用场景：单变量时序预测。需要指定 `fh` 作为预测步长；`compare()` 会比较候选预测器的回测指标。

```python
import numpy as np
import pandas as pd
from happymath.AutoML import TimeSeriesML

series = pd.Series(
    np.sin(np.arange(48) / 4) + np.arange(48) * 0.02,
    index=pd.date_range("2020-01-01", periods=48, freq="MS"),
    name="value",
)

ts = TimeSeriesML(series, fh=3, fold=2, seed=42, verbose=False, html=False)
best_ts = ts.compare(include=["naive", "arima"], sort="MASE", verbose=False)
forecast = ts.predict(estimator=best_ts, fh=3, verbose=False)
print("time_series_best", best_ts.__class__.__name__)
print(forecast)
```

示例输出：

```text
time_series_best ARIMA
         y_pred
2023-10 -0.1839
2023-11 -0.1877
2023-12 -0.2456
```

处理建议：

- 默认主指标 `MASE` 越小越好。
- 有季节性时传入 `seasonal_period`，例如月度年周期可用 `seasonal_period=12`。
- 时序 `scores(mode="kfold")` 复用 PyCaret 回测结果；`scores(mode="holdout")` 需要可对齐的测试集。

## 9. 按问题类型选择接口

| 问题类型                    | 最小接口组合                                                | 说明                                             |
| --------------------------- | ----------------------------------------------------------- | ------------------------------------------------ |
| 分类/回归模型初筛           | `ClassificationML/RegressionML -> compare -> get_results` | 不默认调参，不默认集成。                         |
| 指标导向调参                | `compare/create -> tune -> get_results`                   | `optimize` 与 `sort` 保持一致。              |
| 需要多个模型融合            | `create` 多个模型 -> `blend`                            | 对应 PyCaret`blend_models([m1, m2, ...])`。    |
| 需要两层模型                | `create` 多个模型 -> `stack`                            | 对应 PyCaret`stack_models([m1, m2, ...])`。    |
| 需要单模型 Bagging/Boosting | `create/compare -> ensemble`                              | 对应 PyCaret`ensemble_model(model)`。          |
| 需要泛化评估表              | 当前模型 ->`scores`                                       | 按是否有外部测试集选择`custom/kfold/holdout`。 |
| 聚类                        | `ClusteringML -> create -> assign`                        | 无目标列，不使用分类/回归指标。                  |
| 异常检测                    | `AnomalyML -> create -> assign`                           | 重点设置`fraction`。                           |
| 时序预测                    | `TimeSeriesML -> compare/create -> predict`               | 重点设置`fh/fold/seasonal_period`。            |
