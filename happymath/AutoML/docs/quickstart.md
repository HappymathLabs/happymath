# HappyMath AutoML 快速开始

`happymath.AutoML` 是对 PyCaret 的任务化封装，适合用少量代码完成分类、回归、聚类、异常检测和时序预测。使用时要先明确一个核心原则：`compare()` 只做候选模型横向比较，`tune()` 才做超参数搜索；模型融合、堆叠也不是默认步骤，只有当题目或实验目的明确要求集成学习时再调用。

## 推荐工作流

| 问题目标 | 推荐接口 | 是否需要继续调参或集成 |
|---|---|---|
| 快速判断哪类模型效果较好 | `compare(include=..., sort=...)` | 通常不需要，直接查看排行榜和最佳模型即可。 |
| 已经选定模型，希望优化超参数 | `create(...)` 或 `compare(...)` 后接 `tune(...)` | 需要。调大 `n_iter` 会增加搜索深度和耗时。 |
| 题目明确要求 Bagging / Boosting | `ensemble(...)` | 只对单个基模型做集成，先有 `current_model`。 |
| 题目明确要求投票 / 平均融合 | 至少创建两个模型后 `blend(...)` | 分类可用 `method="soft"` 或 `"hard"`，`auto` 会自动回退。 |
| 题目明确要求 Stacking | 至少创建两个模型后 `stack(...)` | 需要选择或默认使用元模型，耗时高于普通比较。 |
| 需要论文或报告中的泛化评估表 | `scores(mode=..., metrics=...)` | 根据数据量和测试集情况选择 mode。 |
| 需要部署或复现实验结果 | `finalize(...)` 后 `save(...)` | `finalize` 只在全量数据上重新拟合，不会改变超参数。 |

PyCaret 官方示例的基本顺序也是 `setup -> compare_models`，需要调参时才执行 `create_model/compare_models -> tune_model`，需要融合或堆叠时才执行 `blend_models/stack_models`。在 HappyMath 中，`setup` 已经由任务类初始化自动完成。

## compare 和 tune 的关系

- `compare()` 对应 PyCaret 的 `compare_models()`：训练并用交叉验证评估候选模型，按 `sort` 或 `primary_metric` 选择最佳模型。它不会自动调用 `tune_model()`，也不会做轻量调参。
- `tune()` 对应 PyCaret 的 `tune_model()`：对一个已训练模型做超参数搜索，默认 `n_iter=300`（默认迭代次数为 300 次）。这是一个独立步骤，搜索强度主要由 `n_iter`、`custom_grid`、`search_library`、`search_algorithm` 等参数决定。
- 因此，`compare()` 得到最佳模型后是否继续 `tune()` 取决于需求：如果只是快速比较模型效果，停止在 `compare()` 即可；如果需要更充分优化某个入选模型，再调用 `tune(best, n_iter=...)`。
- `tune(choose_better=True)` 默认在调参结果变差时保留输入模型，适合自动脚本；但调参仍可能耗时明显增加。

## 主要类

| 类 | 任务类型 | 默认主指标 |
|---|---|---|
| `ClassificationML` | 分类 | `Accuracy` |
| `RegressionML` | 回归 | `MAE` |
| `ClusteringML` | 聚类 | `Silhouette` |
| `AnomalyML` | 异常检测 | `AUC` |
| `TimeSeriesML` | 时序预测 | `MASE` |

## 分类：只做模型比较

适用场景：题目只要求选择一个分类模型，或你处在建模初筛阶段。此时按照 PyCaret 的 `compare_models()` 思路，只需要输出比较结果，不必自动调参、融合或堆叠。

```python
from sklearn.datasets import load_iris
from happymath.AutoML import ClassificationML

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

best = clf.compare(include=["lr", "dt"], sort="Accuracy", verbose=False)
print("Best:", best.__class__.__name__)
print(clf.get_results()[["Model", "Accuracy", "AUC", "F1"]])
```

运行输出示例：

```text
Best: LogisticRegression
                       Model  Accuracy     AUC      F1
lr       Logistic Regression    0.9667  0.0000  0.9666
dt  Decision Tree Classifier    0.9500  0.9625  0.9499
```

## 回归：比较后按需调参

适用场景：先用 `compare()` 找到较好的回归模型；如果论文或业务目标要求进一步降低误差，再对最佳模型调用 `tune()`。

```python
from sklearn.datasets import load_diabetes
from happymath.AutoML import RegressionML

diabetes = load_diabetes(as_frame=True)
data = diabetes.data.copy()
data["target"] = diabetes.target

reg = RegressionML(
    data=data,
    target="target",
    train_size=0.8,
    fold=2,
    seed=42,
    verbose=False,
    html=False,
)

best = reg.compare(include=["lr", "dt"], sort="MAE", verbose=False)
print("Compare best:", best.__class__.__name__)
print(reg.get_results()[["Model", "MAE", "RMSE", "R2"]])

tuned = reg.tune(estimator=best, n_iter=5, optimize="MAE", verbose=False, tuner_verbose=False)
print("Tuned:", tuned.__class__.__name__)
print(reg.get_results()[["MAE", "RMSE", "R2"]])
```

运行输出示例：

```text
Compare best: LinearRegression
                      Model      MAE     RMSE      R2
lr        Linear Regression  45.6885  56.4420  0.4711
dt  Decision Tree Regressor  66.2042  83.7161 -0.1661
Tuned: LinearRegression
          MAE     RMSE      R2
Mean  45.6885  56.4420  0.4711
```

## 分类：需要概率或阈值时

适用场景：二分类任务更关心召回率、F1、AUC 或预测概率。比较阶段可以改 `sort`；预测阶段可用 `raw_score=True` 输出各类别概率。二分类阈值调整由底层 PyCaret 支持，可通过 `probability_threshold` 透传。

```python
best = clf.compare(include=["lr", "dt"], sort="F1", verbose=False)
pred = clf.predict(
    estimator=best,
    data=data.head(),
    raw_score=True,
    verbose=False,
)
print(pred.filter(regex="target|prediction").head())
```

## 只有明确需要时才融合或堆叠

PyCaret 官方示例会把 `compare_models(n_select=3)` 的结果传入 `blend_models()` 或 `stack_models()`。HappyMath 当前 `compare()` 固定返回一个最佳模型，因此如果需要融合或堆叠，应显式创建多个基模型。

```python
lr = clf.create("lr", verbose=False)
dt = clf.create("dt", verbose=False)
nb = clf.create("nb", verbose=False)

blended = clf.blend(estimator_list=[lr, dt, nb], method="soft", verbose=False)
print("Blended:", blended.__class__.__name__)

stacked = clf.stack(estimator_list=[lr, dt, nb], meta_model_fold=2, verbose=False)
print("Stacked:", stacked.__class__.__name__)
```

如果只是要回答“哪个基础模型效果最好”，不要默认执行 `blend()` 或 `stack()`，否则会增加解释成本和计算成本。

## scores 应该怎么选

`scores()` 用于把当前模型按不同数据切分方式重新整理成评估表，适合报告和论文输出。

| mode | 适用情况 | 输出形式 |
|---|---|---|
| `auto` | 不确定用哪种评估方式 | 有外部测试集时等价 `custom`；监督学习小样本用 `leaveout`，中等样本用 `kfold`，大样本用 `holdout`。 |
| `holdout` | 需要一次训练/测试划分对比 | index 为 `train`、`test`。 |
| `kfold` | 需要 K 折评估表 | 每折一行，最后一行为 `mean`。 |
| `leaveout` | 监督学习极小样本 | index 为 `train_mean`、`test_mean`。 |
| `custom` | 已有独立测试集 | index 为 `train`、`test`；监督学习测试集必须含目标列。 |
| `train-only` | 只想看训练集，或聚类任务 | 至少包含 `train`；若有测试集也会给出 `test`。 |

```python
scores = clf.scores(mode="kfold", metrics=["Accuracy", "F1"], fold=3)
print(scores)
```

## 聚类和异常检测

无监督任务没有 `target`，通常不调用 `compare()`。先 `create()`，再用 `assign()` 把簇标签或异常标签写回数据。

```python
import numpy as np
import pandas as pd
from happymath.AutoML import ClusteringML, AnomalyML

rng = np.random.default_rng(42)
cluster_df = pd.DataFrame(
    np.vstack([
        rng.normal([0, 0], 0.5, size=(30, 2)),
        rng.normal([4, 4], 0.5, size=(30, 2)),
        rng.normal([-4, 4], 0.5, size=(30, 2)),
    ]),
    columns=["x1", "x2"],
)

clu = ClusteringML(cluster_df, seed=42, verbose=False, html=False)
clu.create(model="kmeans", num_clusters=3, verbose=False)
assigned = clu.assign()
print(assigned["Cluster"].value_counts().sort_index())

normal = rng.normal(0, 1, size=(80, 2))
outliers = rng.normal(5, 0.5, size=(8, 2))
anomaly_df = pd.DataFrame(np.vstack([normal, outliers]), columns=["x", "y"])

ano = AnomalyML(anomaly_df, fraction=0.1, seed=42, verbose=False, html=False)
ano.create(model="iforest", verbose=False)
labeled = ano.assign()
print(labeled["Anomaly"].value_counts().sort_index())
```

## 时序预测

时序任务先指定预测步长 `fh`。快速建模时用 `compare()` 选择候选预测器；需要未来预测时调用 `predict(fh=...)`。

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
best = ts.compare(include=["naive", "arima"], sort="MASE", verbose=False)
forecast = ts.predict(estimator=best, fh=3, verbose=False)
print(forecast)
```

## 常见问题处理

- LightGBM、XGBoost、CatBoost 或 Optuna 等可选依赖不可用时，先用 `include=[...]` 限定基础模型，例如 `["lr", "dt", "rf"]`；需要特定算法时再安装对应依赖。
- 调参结果不如默认模型时，保留 `choose_better=True`；需要研究调参过程时，可降低 `n_iter` 快速试跑，再逐步增大。
- 指标不符合题目目标时，分类用 `sort="F1"`、`sort="AUC"` 等，回归用 `sort="MAE"`、`sort="RMSE"`、`sort="R2"` 等；`tune(optimize=...)` 要与 `compare(sort=...)` 保持一致。
- 预测新数据时，监督学习的新数据可以不含目标列；做 `scores(custom)` 时，测试集必须含目标列才能计算指标。
- 最终提交模型前先 `finalize()`，再 `save()`；`finalize()` 只是用全量数据重新拟合当前模型，不会重新比较模型，也不会自动调参。
