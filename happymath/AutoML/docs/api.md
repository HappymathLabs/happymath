# HappyMath AutoML API 文档

本页按类整理 `happymath.AutoML` 模块的公开接口，所有参数含义均来自源码与 PyCaret 封装逻辑，未经验证的信息不予列出。

## 目录

- [AutoMLBase](#automlbase)
- [ClassificationML](#classificationml)
- [RegressionML](#regressionml)
- [ClusteringML](#clusteringml)
- [AnomalyML](#anomalyml)
- [TimeSeriesML](#timeseriesml)

---

## AutoMLBase

所有任务类的公共基类，封装了 PyCaret experiment 生命周期、模型存储、指标处理与通用训练接口。通常不应直接实例化，而是通过 `ClassificationML`、`RegressionML` 等子类使用。

### 构造函数

```python
AutoMLBase(
    data: DataLike,
    target: TargetLike = None,
    test_data: Optional[DataLike] = None,
    primary_metric: Optional[str] = None,
    **setup_kwargs: Any,
)
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `data` | `DataLike` | 必填 | 训练数据，通常为 pandas DataFrame。 |
| `target` | `TargetLike` | `None` | 目标列名称；聚类、异常检测任务不传。 |
| `test_data` | `Optional[DataLike]` | `None` | 外部测试集，不传时使用 PyCaret 内部分割。 |
| `primary_metric` | `Optional[str]` | `None` | 主指标，用于 `get_best_model` 与优化方向判断。 |
| `**setup_kwargs` | `Any` | - | 透传给 PyCaret `setup()` 的额外参数。 |

### 通用属性

| 属性 | 类型 | 说明 |
|---|---|---|
| `data` | `pd.DataFrame` | 加载并校验后的训练数据。 |
| `target` | `Optional[str]` | 目标列名称。 |
| `test_data` | `Optional[pd.DataFrame]` | 外部测试集（若有）。 |
| `primary_metric` | `Optional[str]` | 当前任务的主指标。 |
| `setup_kwargs` | `Dict[str, Any]` | 初始化时传入的额外 setup 参数。 |
| `experiment` | PyCaret Experiment | 底层 PyCaret 实验对象。 |
| `current_model` | `Optional[Any]` | 当前选中的模型对象。 |
| `results` | `Optional[Any]` | 最近一次 `pull()` 得到的结果表。 |
| `is_setup` | `bool` | 实验是否已完成 setup。 |
| `models` | `Dict[str, StoredModel]` | 已存储模型的字典（向后兼容属性）。 |

### compare

```python
compare(
    include: Optional[List[Any]] = None,
    exclude: Optional[List[str]] = None,
    sort: Optional[str] = None,
    budget_time: Optional[float] = None,
    verbose: Optional[bool] = None,
    **kwargs: Any,
) -> Any
```

比较多个基线模型并返回表现最好的模型。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `include` | `Optional[List[Any]]` | `None` | 指定参与对比的模型 ID 列表，例如 `["lr", "dt"]`。 |
| `exclude` | `Optional[List[str]]` | `None` | 排除的模型 ID 列表。 |
| `sort` | `Optional[str]` | `None` | 排序指标；默认使用 `self.primary_metric`。 |
| `budget_time` | `Optional[float]` | `None` | 单模型最大训练时间（秒），用于 `budget_time` 模式。 |
| `verbose` | `Optional[bool]` | `None` | 是否打印详细输出；默认使用实例的 `verbose`。 |
| `**kwargs` | `Any` | - | 透传给 PyCaret `compare_models`。 |

### create

```python
create(
    estimator: Any,
    return_train_score: bool = False,
    verbose: Optional[bool] = None,
    **kwargs: Any,
) -> Any
```

使用指定算法创建并训练一个模型。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `estimator` | `Any` | 必填 | 模型 ID（如 `"lr"`）或模型对象。 |
| `return_train_score` | `bool` | `False` | 是否返回训练集得分。 |
| `verbose` | `Optional[bool]` | `None` | 是否打印详细输出。 |
| `**kwargs` | `Any` | - | 透传给 PyCaret `create_model`。 |

### tune

```python
tune(
    estimator: Optional[Any] = None,
    n_iter: int = 10,
    custom_grid: Optional[Dict[str, List[Any]]] = None,
    optimize: Optional[str] = None,
    verbose: Optional[bool] = None,
    tuner_verbose: Union[int, bool] = True,
    choose_better: bool = True,
    **kwargs: Any,
) -> Any
```

对当前模型或指定模型进行超参数调优。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `estimator` | `Optional[Any]` | `None` | 待调优模型；默认使用 `current_model`。 |
| `n_iter` | `int` | `10` | 搜索迭代次数。 |
| `custom_grid` | `Optional[Dict[str, List[Any]]]` | `None` | 自定义参数网格。 |
| `optimize` | `Optional[str]` | `None` | 优化指标；默认使用 `primary_metric`。 |
| `verbose` | `Optional[bool]` | `None` | 是否打印详细输出。 |
| `tuner_verbose` | `Union[int, bool]` | `True` | 调优器本身的输出级别。 |
| `choose_better` | `bool` | `True` | 若调优后效果变差，是否保留原模型。 |
| `**kwargs` | `Any` | - | 透传给 PyCaret `tune_model`。 |

### ensemble

```python
ensemble(
    estimator: Optional[Any] = None,
    method: str = "Bagging",
    n_estimators: int = 10,
    optimize: Optional[str] = None,
    verbose: Optional[bool] = None,
    **kwargs: Any,
) -> Any
```

对基模型做 Bagging 或 Boosting 集成。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `estimator` | `Optional[Any]` | `None` | 基模型；默认使用 `current_model`。 |
| `method` | `str` | `"Bagging"` | 集成方法，`"Bagging"` 或 `"Boosting"`（受 PyCaret 版本限制）。 |
| `n_estimators` | `int` | `10` | 集成中基学习器数量。 |
| `optimize` | `Optional[str]` | `None` | 优化指标；默认使用 `primary_metric`。 |
| `verbose` | `Optional[bool]` | `None` | 是否打印详细输出。 |
| `**kwargs` | `Any` | - | 透传给 PyCaret `ensemble_model`。 |

### blend

```python
blend(
    estimator_list: Optional[List[Any]] = None,
    optimize: Optional[str] = None,
    method: str = "auto",
    weights: Optional[List[float]] = None,
    verbose: Optional[bool] = None,
    **kwargs: Any,
) -> Any
```

通过投票（分类）或平均（回归）方式融合多个模型。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `estimator_list` | `Optional[List[Any]]` | `None` | 待融合模型列表；默认使用内部存储的全部模型。 |
| `optimize` | `Optional[str]` | `None` | 优化指标；默认使用 `primary_metric`。 |
| `method` | `str` | `"auto"` | 融合方法，如 `"auto"`、`"soft"`、`"hard"` 等。 |
| `weights` | `Optional[List[float]]` | `None` | 各模型投票权重。 |
| `verbose` | `Optional[bool]` | `None` | 是否打印详细输出。 |
| `**kwargs` | `Any` | - | 透传给 PyCaret `blend_models`。 |

### stack

```python
stack(
    estimator_list: Optional[List[Any]] = None,
    meta_model: Optional[Any] = None,
    meta_model_fold: Optional[int] = 5,
    method: str = "auto",
    restack: bool = False,
    optimize: Optional[str] = None,
    verbose: Optional[bool] = None,
    **kwargs: Any,
) -> Any
```

构建两层 Stacking 集成。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `estimator_list` | `Optional[List[Any]]` | `None` | 第一层基模型列表；默认使用内部存储的全部模型。 |
| `meta_model` | `Optional[Any]` | `None` | 第二层元模型；默认使用 PyCaret 推荐模型。 |
| `meta_model_fold` | `Optional[int]` | `5` | 元模型训练时的折数。 |
| `method` | `str` | `"auto"` | Stacking 方法。 |
| `restack` | `bool` | `False` | 是否在最终预测中使用原始特征与基模型预测共同输入。 |
| `optimize` | `Optional[str]` | `None` | 优化指标；默认使用 `primary_metric`。 |
| `verbose` | `Optional[bool]` | `None` | 是否打印详细输出。 |
| `**kwargs` | `Any` | - | 透传给 PyCaret `stack_models`。 |

### predict

```python
predict(
    estimator: Optional[Any] = None,
    data: Optional[pd.DataFrame] = None,
    raw_score: bool = False,
    verbose: Optional[bool] = None,
    **kwargs: Any,
) -> pd.DataFrame
```

使用指定模型或当前模型进行预测。若未提供 `data`，则使用 `self.test_data`。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `estimator` | `Optional[Any]` | `None` | 用于预测的模型；默认使用 `current_model`。 |
| `data` | `Optional[pd.DataFrame]` | `None` | 待预测数据；默认使用测试集。 |
| `raw_score` | `bool` | `False` | 是否输出原始概率分；仅在分类等支持的任务中生效。 |
| `verbose` | `Optional[bool]` | `None` | 是否打印详细输出。 |
| `**kwargs` | `Any` | - | 透传给 PyCaret `predict_model`。 |

### finalize

```python
finalize(estimator: Optional[Any] = None) -> Any
```

在全量数据上重新训练模型，得到最终部署模型。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `estimator` | `Optional[Any]` | `None` | 待 finalize 的模型；默认使用 `current_model`。 |

### evaluate

```python
evaluate(estimator: Optional[Any] = None) -> None
```

启动 PyCaret 交互式评估 UI（在 Notebook 中可用）。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `estimator` | `Optional[Any]` | `None` | 待评估模型；默认使用 `current_model`。 |

### plot

```python
plot(
    estimator: Optional[Any] = None,
    plot_type: str = "auc",
    scale: float = 1.0,
    save: bool = False,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_title: Optional[str] = None,
    legend_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    plot_kwargs: Optional[Dict[str, Any]] = None,
    font_sizes: Optional[Dict[str, Union[int, float]]] = None,
    verbose: Optional[bool] = None,
) -> Optional[str]
```

调用 PyCaret 绘图，并支持中文标题等自定义。`plot_type="tree"` 与 `"tree_text"` 为决策树专属实现。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `estimator` | `Optional[Any]` | `None` | 待绘图模型；默认使用 `current_model`。 |
| `plot_type` | `str` | `"auc"` | 图表类型。 |
| `scale` | `float` | `1.0` | 图像缩放比例。 |
| `save` | `bool` | `False` | 是否保存图像；决策树可视化中可传入路径字符串。 |
| `title` | `Optional[str]` | `None` | 图表标题。 |
| `xlabel` | `Optional[str]` | `None` | X 轴标签。 |
| `ylabel` | `Optional[str]` | `None` | Y 轴标签。 |
| `legend_title` | `Optional[str]` | `None` | 图例标题。 |
| `legend_labels` | `Optional[List[str]]` | `None` | 图例标签。 |
| `figsize` | `Tuple[int, int]` | `(10, 6)` | 图像尺寸。 |
| `plot_kwargs` | `Optional[Dict[str, Any]]` | `None` | 透传给 PyCaret plot 的额外参数。 |
| `font_sizes` | `Optional[Dict[str, Union[int, float]]]` | `None` | 各文字元素字号。 |
| `verbose` | `Optional[bool]` | `None` | 是否打印详细输出。 |

### scores

```python
scores(
    mode: str = "auto",
    metrics: Union[str, List[str]] = "all",
    test_data: Optional[DataLike] = None,
    train_size: Optional[float] = None,
    fold: Optional[int] = None,
) -> pd.DataFrame
```

使用当前模型按不同切分方式评估性能，返回 DataFrame。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `mode` | `str` | `"auto"` | 评估模式：`auto`、`holdout`、`kfold`、`leaveout`、`custom`、`train-only`。 |
| `metrics` | `Union[str, List[str]]` | `"all"` | 指标选择：`"all"`、单个指标名或指标名列表。 |
| `test_data` | `Optional[DataLike]` | `None` | 自定义测试集；仅在 `custom` / `train-only` 模式下使用。 |
| `train_size` | `Optional[float]` | `None` | `holdout` 模式训练集比例；默认依次为传入值、`setup_kwargs` 中的 `train_size`、0.7。 |
| `fold` | `Optional[int]` | `None` | `kfold` 模式折数；默认依次为传入值、`setup_kwargs` 中的 `fold`、5。 |

### get_best_model

```python
get_best_model() -> Tuple[Any, Dict[str, Any]]
```

根据 `primary_metric` 从已保存模型中挑选最优模型，返回 `(model, metrics)` 二元组。

### get_results

```python
get_results() -> pd.DataFrame
```

获取最近一次结果表；若不存在则调用 PyCaret `pull()`。

### get_leaderboard

```python
get_leaderboard() -> pd.DataFrame
```

获取模型排行榜。优先调用 PyCaret `get_leaderboard()`，否则回退到 `results` 或 `pull()`。

### get_metrics

```python
get_metrics() -> Any
```

返回当前任务支持的指标列表。通常返回 pandas DataFrame，列包含 `ID` 与 `Display Name`。

### get_models

```python
get_models() -> Iterable[str]
```

返回已存储模型的名称列表。

### save

```python
save(model_name: str, model: Optional[Any] = None) -> None
```

将模型保存到磁盘。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model_name` | `str` | 必填 | 保存路径或文件名。 |
| `model` | `Optional[Any]` | `None` | 待保存模型；默认使用 `current_model`。 |

### load

```python
load(model_name: str) -> Any
```

从磁盘加载模型。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model_name` | `str` | 必填 | 保存路径或文件名。 |

### get_config

```python
get_config(key: Optional[str] = None) -> Any
```

读取 PyCaret 实验配置。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `key` | `Optional[str]` | `None` | 配置项名称；`None` 返回全部配置。 |

---

## ClassificationML

```python
ClassificationML(
    data: Any,
    target: Any = None,
    test_data: Optional[Any] = None,
    train_size: float = 0.7,
    fold: int = 5,
    seed: int = 42,
    n_jobs: int = -1,
    verbose: bool = False,
    html: bool = False,
    primary_metric: Optional[str] = None,
    **setup_kwargs: Any,
)
```

分类任务封装类，继承自 `AutoMLBase`。默认主指标为 `Accuracy`。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `data` | `Any` | 必填 | 训练数据。 |
| `target` | `Any` | `None` | 目标列。 |
| `test_data` | `Optional[Any]` | `None` | 外部测试集。 |
| `train_size` | `float` | `0.7` | 训练集比例。 |
| `fold` | `int` | `5` | 交叉验证折数。 |
| `seed` | `int` | `42` | 随机种子。 |
| `n_jobs` | `int` | `-1` | 并行作业数。 |
| `verbose` | `bool` | `False` | 是否输出详细日志。 |
| `html` | `bool` | `False` | 是否启用 PyCaret HTML 输出。 |
| `primary_metric` | `Optional[str]` | `None` | 主指标，默认 `Accuracy`。 |
| `**setup_kwargs` | `Any` | - | 透传给 `pycaret.classification.setup`。 |

### 方法

继承 `AutoMLBase` 的全部方法。`plot` 默认 `plot_type="auc"`；若传入非典型回归图类型，`RegressionML.plot` 会给出提示，而 `ClassificationML` 直接使用基类 `plot`。

---

## RegressionML

```python
RegressionML(
    data: Any,
    target: Any = None,
    test_data: Optional[Any] = None,
    train_size: float = 0.7,
    fold: int = 5,
    seed: int = 42,
    n_jobs: int = -1,
    verbose: bool = False,
    html: bool = False,
    primary_metric: Optional[str] = None,
    **setup_kwargs: Any,
)
```

回归任务封装类，继承自 `AutoMLBase`。默认主指标为 `MAE`。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `data` | `Any` | 必填 | 训练数据。 |
| `target` | `Any` | `None` | 目标列。 |
| `test_data` / `train_size` / `fold` / `seed` / `n_jobs` / `verbose` / `html` | 同上 | 同上 | 同上。 |
| `primary_metric` | `Optional[str]` | `None` | 主指标，默认 `MAE`。 |
| `**setup_kwargs` | `Any` | - | 透传给 `pycaret.regression.setup`。 |

### plot

重写了基类 `plot`，默认 `plot_type="residuals"`，并在传入非回归典型图类型时打印警告。支持的回归图类型包括：`residuals`、`error`、`cooks`、`feature`、`learning`、`tree`、`tree_text`。

---

## ClusteringML

```python
ClusteringML(
    data: Any,
    test_data: Optional[Any] = None,
    seed: int = 42,
    n_jobs: int = -1,
    verbose: bool = False,
    html: bool = False,
    primary_metric: Optional[str] = None,
    **setup_kwargs: Any,
)
```

聚类任务封装类，继承自 `AutoMLBase`。默认主指标为 `Silhouette`，不需要 `target`。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `data` | `Any` | 必填 | 训练数据。 |
| `test_data` | `Optional[Any]` | `None` | 外部测试集。 |
| `seed` / `n_jobs` / `verbose` / `html` | 同上 | 同上 | 同上。 |
| `primary_metric` | `Optional[str]` | `None` | 主指标，默认 `Silhouette`。 |
| `**setup_kwargs` | `Any` | - | 透传给 `pycaret.clustering.setup`。 |

### create

```python
create(
    model: str = "kmeans",
    num_clusters: int = 4,
    verbose: Optional[bool] = None,
    **kwargs: Any,
) -> Any
```

创建聚类模型。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | `str` | `"kmeans"` | 聚类算法 ID，如 `"kmeans"`、`"hclust"`、`"dbscan"` 等。 |
| `num_clusters` | `int` | `4` | 聚类簇数。 |
| `verbose` | `Optional[bool]` | `None` | 是否输出详细日志。 |
| `**kwargs` | `Any` | - | 透传给 PyCaret `create_model`。 |

### assign

```python
assign(model: Optional[Any] = None)
```

为数据分配簇标签。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | `Optional[Any]` | `None` | 聚类模型；默认使用 `current_model`。 |

返回的 DataFrame 会新增 `Cluster` 列。

---

## AnomalyML

```python
AnomalyML(
    data: Any,
    test_data: Optional[Any] = None,
    fraction: float = 0.05,
    seed: int = 42,
    n_jobs: int = -1,
    verbose: bool = False,
    html: bool = False,
    primary_metric: Optional[str] = None,
    **setup_kwargs: Any,
)
```

异常检测任务封装类，继承自 `AutoMLBase`。默认主指标为 `AUC`，不需要 `target`。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `data` | `Any` | 必填 | 训练数据。 |
| `test_data` | `Optional[Any]` | `None` | 外部测试集。 |
| `fraction` | `float` | `0.05` | 预期异常样本比例。 |
| `seed` / `n_jobs` / `verbose` / `html` | 同上 | 同上 | 同上。 |
| `primary_metric` | `Optional[str]` | `None` | 主指标，默认 `AUC`。 |
| `**setup_kwargs` | `Any` | - | 透传给 `pycaret.anomaly.setup`。 |

### create

```python
create(
    model: str = "iforest",
    fraction: Optional[float] = None,
    verbose: Optional[bool] = None,
    **kwargs: Any,
) -> Any
```

创建异常检测模型。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | `str` | `"iforest"` | 异常检测算法 ID，如 `"iforest"`、`"lof"`、`"abod"` 等。 |
| `fraction` | `Optional[float]` | `None` | 异常比例；默认使用初始化时的 `self.fraction`。 |
| `verbose` | `Optional[bool]` | `None` | 是否输出详细日志。 |
| `**kwargs` | `Any` | - | 透传给 PyCaret `create_model`。 |

### assign

```python
assign(model: Optional[Any] = None)
```

为数据标记异常。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | `Optional[Any]` | `None` | 异常检测模型；默认使用 `current_model`。 |

返回的 DataFrame 会新增 `Anomaly` 与 `Anomaly_Score` 列。

---

## TimeSeriesML

```python
TimeSeriesML(
    data: Any,
    target: Any = None,
    test_data: Optional[Any] = None,
    fh: int = 12,
    fold: int = 3,
    seasonal_period: Optional[int] = None,
    seed: int = 42,
    n_jobs: int = -1,
    verbose: bool = False,
    html: bool = False,
    primary_metric: Optional[str] = None,
    **setup_kwargs: Any,
)
```

时序预测任务封装类，继承自 `AutoMLBase`。默认主指标为 `MASE`。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `data` | `Any` | 必填 | 训练数据，通常需要日期索引或单变量序列。 |
| `target` | `Any` | `None` | 目标列名称。 |
| `test_data` | `Optional[Any]` | `None` | 外部测试集。 |
| `fh` | `int` | `12` | 预测步长（forecast horizon）。 |
| `fold` | `int` | `3` | 交叉验证折数。 |
| `seasonal_period` | `Optional[int]` | `None` | 季节周期，例如月度数据可设为 `12`。 |
| `seed` / `n_jobs` / `verbose` / `html` | 同上 | 同上 | 同上。 |
| `primary_metric` | `Optional[str]` | `None` | 主指标，默认 `MASE`。 |
| `**setup_kwargs` | `Any` | - | 透传给 `pycaret.time_series.setup`。 |

### predict

```python
predict(
    estimator: Optional[Any] = None,
    fh: Optional[int] = None,
    X: Optional[Any] = None,
    return_pred_int: bool = False,
    verbose: Optional[bool] = None,
    **kwargs: Any,
)
```

时序预测。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `estimator` | `Optional[Any]` | `None` | 用于预测的模型；默认使用 `current_model`。 |
| `fh` | `Optional[int]` | `None` | 预测步长；默认使用初始化时的 `self.fh`。 |
| `X` | `Optional[Any]` | `None` | 外生变量。 |
| `return_pred_int` | `bool` | `False` | 是否返回预测区间。 |
| `verbose` | `Optional[bool]` | `None` | 是否输出详细日志。 |
| `**kwargs` | `Any` | - | 透传给 PyCaret `predict_model`。 |

> 注意：`TimeSeriesML` 继承自 `AutoMLBase`，因此同样可以使用 `compare`、`create`、`tune`、`scores` 等通用接口，但时序任务对数据格式与 PyCaret 版本较为敏感，建议先在小样本数据上验证路径可行性。
