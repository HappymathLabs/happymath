# HappyMath AutoML 快速开始

`happymath.AutoML` 是基于 [PyCaret](https://pycaret.org/) 封装的一套低代码自动化机器学习接口，覆盖分类、回归、聚类、异常检测和时序预测等常见任务。它的目标是让你用几行代码完成数据准备、模型训练、调优、集成、评估与保存。

## 主要类

| 类 | 任务类型 | 默认主指标 |
|---|---|---|
| `ClassificationML` | 分类 | Accuracy |
| `RegressionML` | 回归 | MAE |
| `ClusteringML` | 聚类 | Silhouette |
| `AnomalyML` | 异常检测 | AUC |
| `TimeSeriesML` | 时序预测 | MASE |
| `AutoMLBase` | 所有任务公共基类 | 由子类决定 |

## 常用方法

- `compare(...)`：比较多个基线模型，返回表现最好的模型。
- `create(estimator, ...)`：使用指定算法创建模型。
- `tune(estimator, n_iter, ...)`：对模型做超参数调优。
- `ensemble(method, n_estimators, ...)`：对当前模型做 Bagging / Boosting 集成。
- `blend(...)`：将多个模型按投票 / 平均方式融合。
- `stack(...)`：构建两层 Stacking 集成。
- `predict(data, ...)`：使用当前或指定模型进行预测。
- `get_best_model()`：从已训练模型中挑选主指标最优的模型。
- `scores(mode, metrics, ...)`：按指定切分方式评估当前模型。
- `save(model_name, model)` / `load(model_name)`：保存 / 加载模型。
- `get_results()` / `get_leaderboard()` / `get_metrics()`：获取结果表、排行榜和指标列表。
- `plot(plot_type, ...)`：调用 PyCaret 绘图，支持中文标题。

## 关键参数说明

### `ClassificationML` / `RegressionML`

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `data` | Any | 必填 | 训练数据，支持 pandas DataFrame 等格式。 |
| `target` | Any | `None` | 目标列名称。 |
| `test_data` | Optional[Any] | `None` | 外部测试集。 |
| `train_size` | float | `0.7` | 训练集比例。 |
| `fold` | int | `5` | 交叉验证折数。 |
| `seed` | int | `42` | 随机种子，对应 PyCaret 的 `session_id`。 |
| `n_jobs` | int | `-1` | 并行作业数。 |
| `verbose` | bool | `False` | 是否输出详细日志。 |
| `html` | bool | `False` | 是否启用 PyCaret 的 HTML 输出。 |
| `primary_metric` | Optional[str] | `None` | 主指标，分类默认 `Accuracy`，回归默认 `MAE`。 |

### `ClusteringML`

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `data` | Any | 必填 | 训练数据。 |
| `test_data` | Optional[Any] | `None` | 外部测试集。 |
| `seed` | int | `42` | 随机种子。 |
| `n_jobs` | int | `-1` | 并行作业数。 |
| `verbose` | bool | `False` | 是否输出详细日志。 |
| `html` | bool | `False` | 是否启用 HTML 输出。 |
| `primary_metric` | Optional[str] | `None` | 默认 `Silhouette`。 |

### `AnomalyML`

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `data` | Any | 必填 | 训练数据。 |
| `test_data` | Optional[Any] | `None` | 外部测试集。 |
| `fraction` | float | `0.05` | 异常样本预期比例。 |
| `seed` / `n_jobs` / `verbose` / `html` | 同上 | 同上 | 同上。 |
| `primary_metric` | Optional[str] | `None` | 默认 `AUC`。 |

### `TimeSeriesML`

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `data` | Any | 必填 | 训练数据。 |
| `target` | Any | `None` | 目标列名称。 |
| `test_data` | Optional[Any] | `None` | 外部测试集。 |
| `fh` | int | `12` | 预测步长（forecast horizon）。 |
| `fold` | int | `3` | 交叉验证折数。 |
| `seasonal_period` | Optional[int] | `None` | 季节周期。 |
| `seed` / `n_jobs` / `verbose` / `html` | 同上 | 同上 | 同上。 |
| `primary_metric` | Optional[str] | `None` | 默认 `MASE`。 |

## ⚠️ 已知限制与注意事项

1. **PyCaret 首次运行**：首次调用任一任务类时，PyCaret 会进行环境初始化与依赖检查，耗时可能较长（数十秒到数分钟），请耐心等待。
2. **LightGBM 依赖缺失**：如果环境未安装 `lightgbm`，在 `compare` 默认包含 LightGBM 相关模型时会跳过或报错。建议显式通过 `include=[...]` 限制模型列表，例如 `include=["lr", "dt"]`。
3. **时序预测（`TimeSeriesML`）**：当前实现较为基础，`predict` 接口与 PyCaret time_series 模块深度绑定，部分模型或参数组合可能出现兼容性问题。建议在简单场景下使用，并优先验证可行路径。
4. **绘图（`plot`）**：`plot_type="tree"` 需要系统安装 Graphviz 可执行文件以及 `graphviz` Python 包；其他部分 plot 类型依赖 PyCaret 版本，可能在某些环境下失效。
5. **CatBoost**：模块已设置 `CATBOOST_ALLOW_WRITING_FILES=0` 以避免生成临时文件，但在某些系统上仍可能出现权限或版本相关警告。
6. **集成与融合**：`blend` / `stack` 要求内部至少保存了两个模型；`ensemble` 仅对单个基模型生效，具体支持的 `method` 取决于 PyCaret 对应任务模块。

## 最小可运行示例：Iris 分类

```python
from sklearn.datasets import load_iris
from happymath.AutoML import ClassificationML

# 加载数据
iris = load_iris(as_frame=True)
data = iris.data.copy()
data["target"] = iris.target

# 初始化并自动完成 PyCaret setup
clf = ClassificationML(
    data=data,
    target="target",
    train_size=0.8,
    fold=2,
    seed=42,
    verbose=False,
    html=False,
)

# 比较模型（仅比较逻辑回归与决策树，避免 LightGBM 等依赖问题）
best = clf.compare(include=["lr", "dt"], sort="Accuracy", verbose=False)
print("Best model:", best)

# 预测
predictions = clf.predict(data=data.head())
print(predictions[["target", "prediction_label"]])

# 获取最优模型
model, metrics = clf.get_best_model()
print("Metrics:", metrics)
```
