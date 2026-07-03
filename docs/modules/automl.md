# AutoML 自动化机器学习

`happymath.AutoML` 模块基于 PyCaret 提供了一套统一的自动化机器学习接口，覆盖分类、回归、聚类、异常检测和时序预测任务。

## 模块结构

| 类名 | 任务类型 |
|------|----------|
| `ClassificationML` | 分类 |
| `RegressionML` | 回归 |
| `ClusteringML` | 聚类 |
| `AnomalyML` | 异常检测 |
| `TimeSeriesML` | 时序预测 |
| `AutoMLBase` | 所有任务的公共基类 |

## 公共接口

所有任务类都继承自 `AutoMLBase`，提供以下通用方法：

- `compare(...)`：比较多个模型并返回最优模型
- `create(estimator, ...)`：创建指定算法模型
- `tune(estimator, ...)`：超参数调优
- `ensemble(...)` / `blend(...)` / `stack(...)`：模型集成
- `predict(data, ...)`：使用当前模型预测
- `get_best_model()`：获取当前最优模型及指标
- `get_results()` / `get_leaderboard()`：获取实验结果表
- `scores(mode, ...)`：模型评估
- `save(name)` / `load(name)`：模型保存与加载

## 分类任务示例

```python
from happymath.AutoML import ClassificationML
from sklearn.datasets import load_iris
import pandas as pd

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

# 创建并比较模型
lr = clf.create("lr", verbose=False)
best = clf.compare(include=["lr", "dt"], sort="Accuracy", verbose=False)

# 调优与融合
tuned = clf.tune(estimator=lr, n_iter=2, verbose=False)
blended = clf.blend(method="soft", verbose=False)

# 预测
pred = clf.predict(data=data.head())
print(pred[["target", "prediction_label", "prediction_score"]])

# 获取最优模型
model, metrics = clf.get_best_model()
print(metrics)
```

## 回归任务示例

```python
from happymath.AutoML import RegressionML
from sklearn.datasets import load_diabetes

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
ensemble = reg.ensemble(method="Bagging", n_estimators=5, verbose=False)
pred = reg.predict(data=data.head())
print(pred[["target", "prediction_label"]].head())
```

## 聚类任务示例

```python
import numpy as np
import pandas as pd
from happymath.AutoML import ClusteringML

rng = np.random.default_rng(42)
centers = np.array([[0, 0], [5, 5], [-5, 5]])
samples = [center + rng.normal(scale=0.5, size=(60, 2)) for center in centers]
data = pd.DataFrame(np.vstack(samples), columns=["x1", "x2"])

clu = ClusteringML(data=data, seed=42, verbose=False, html=False)
km = clu.create(model="kmeans", num_clusters=3, verbose=False)
assigned = clu.assign()
print(assigned.head())
```

## 异常检测示例

```python
import numpy as np
import pandas as pd
from happymath.AutoML import AnomalyML

rng = np.random.default_rng(0)
normal = rng.normal(0, 1, size=(120, 2))
anomalies = rng.normal(5, 0.5, size=(10, 2))
data = pd.DataFrame(np.vstack([normal, anomalies]), columns=["x", "y"])

ano = AnomalyML(data=data, fraction=0.1, seed=42, verbose=False, html=False)
iforest = ano.create(model="iforest", verbose=False)
labeled = ano.assign()
print(labeled[["Anomaly", "Anomaly_Score"]].head())
```

## 关键参数说明

- `data`：训练数据，支持 pandas DataFrame 或类似结构
- `target`：目标列名（分类/回归/时序需要）
- `train_size`：训练集比例，默认 0.7
- `fold`：交叉验证折数
- `seed`：随机种子，保证可复现
- `verbose` / `html`：控制 PyCaret 输出与 HTML 日志

## 注意事项

- 首次运行可能会下载模型依赖或编译组件，耗时较长
- 聚类和异常检测任务不需要 `target` 参数
- 时序任务需要额外指定 `fh`（预测步长）
- HappyMath 已在内部禁用 PyCaret 和 CatBoost 的日志文件生成，避免污染工作目录
