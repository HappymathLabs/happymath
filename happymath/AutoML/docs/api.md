# HappyMath AutoML API 文档

本页说明 `happymath.AutoML` 的公开接口、设计思想、参数功能和输出形式。AutoML 模块底层使用 PyCaret，HappyMath 负责统一数据载入、任务初始化、模型存储、结果读取和评估表生成。

## 核心设计

- `ClassificationML`、`RegressionML`、`ClusteringML`、`AnomalyML`、`TimeSeriesML` 在初始化时自动完成 PyCaret `setup()`。
- `compare()` 对应 PyCaret `compare_models()`，用于横向比较候选模型，不会自动调用 `tune_model()`，也不存在内置“轻量调参”。
- `tune()` 对应 PyCaret `tune_model()`，用于对一个已训练模型进行超参数搜索。默认 `n_iter=10`，搜索越深耗时越长。
- `ensemble()`、`blend()`、`stack()` 是高级集成接口，不是基础比较流程的默认步骤。
- 最近一次 PyCaret 评分表保存在 `results`，可通过 `get_results()` 读取；已训练模型会存入 `models`，可通过 `get_best_model()` 按主指标选择。

## 数据输入

`data` 支持以下形式：

| 类型 | 说明 |
|---|---|
| `pandas.DataFrame` | 推荐形式；监督学习需包含目标列。 |
| `pandas.Series` | 常用于单变量时序；会转成一列 DataFrame。 |
| `.csv` / `.xlsx` / `.xls` 路径 | 自动读取为 DataFrame。 |
| `numpy.ndarray` | 自动生成 `feature_0`、`feature_1` 等列名；监督学习可用整数 `target` 指定目标列。 |
| `(X, y)` 元组 | 自动合并为 DataFrame，目标列默认为 `target`。 |
| sklearn Bunch | 如 `load_iris(as_frame=True)` 返回对象；未指定目标列时自动命名为 `target`。 |

## 任务类

### ClassificationML

```python
ClassificationML(
    data,
    target=None,
    test_data=None,
    train_size=0.7,
    fold=5,
    seed=42,
    n_jobs=-1,
    verbose=False,
    html=False,
    primary_metric=None,
    **setup_kwargs,
)
```

用于分类任务。默认主指标为 `Accuracy`。

输出：实例化后返回 `ClassificationML` 对象，内部已完成 PyCaret `ClassificationExperiment.setup()`。后续模型训练接口输出分类器或 sklearn pipeline，预测输出通常包含原特征、目标列（若输入中存在）、`prediction_label` 和 `prediction_score` / `prediction_score_*`。

### RegressionML

构造参数与 `ClassificationML` 基本一致。用于回归任务，默认主指标为 `MAE`。

输出：实例化后返回 `RegressionML` 对象，内部已完成 PyCaret `RegressionExperiment.setup()`。预测输出通常包含原特征、目标列（若输入中存在）和 `prediction_label`。

### ClusteringML

```python
ClusteringML(
    data,
    test_data=None,
    seed=42,
    n_jobs=-1,
    verbose=False,
    html=False,
    primary_metric=None,
    **setup_kwargs,
)
```

用于聚类任务，无 `target`。默认主指标为 `Silhouette`。

输出：实例化后返回 `ClusteringML` 对象。`create()` 输出聚类模型，`assign()` 输出带 `Cluster` 列的 DataFrame。

### AnomalyML

```python
AnomalyML(
    data,
    test_data=None,
    fraction=0.05,
    seed=42,
    n_jobs=-1,
    verbose=False,
    html=False,
    primary_metric=None,
    **setup_kwargs,
)
```

用于异常检测任务，无 `target`。`fraction` 表示预期异常比例。

输出：实例化后返回 `AnomalyML` 对象。`create()` 输出异常检测模型，`assign()` 输出带 `Anomaly` 和 `Anomaly_Score` 列的 DataFrame。

### TimeSeriesML

```python
TimeSeriesML(
    data,
    target=None,
    test_data=None,
    fh=12,
    fold=3,
    seasonal_period=None,
    seed=42,
    n_jobs=-1,
    verbose=False,
    html=False,
    primary_metric=None,
    **setup_kwargs,
)
```

用于时序预测任务。默认主指标为 `MASE`。

输出：实例化后返回 `TimeSeriesML` 对象。`compare()` / `create()` 输出时序预测器或 pipeline，`predict()` 输出包含 `y_pred` 的 DataFrame；请求预测区间时可能额外包含区间列。

## 通用属性

| 属性 | 类型 | 含义 |
|---|---|---|
| `data` | `pd.DataFrame` | 规范化后的训练数据。 |
| `target` | `Optional[str]` | 目标列名；无监督任务为 `None`。 |
| `test_data` | `Optional[pd.DataFrame]` | 外部测试集。 |
| `primary_metric` | `Optional[str]` | 主指标，用于排序和最优模型选择。 |
| `experiment` | PyCaret Experiment | 底层 PyCaret 实验对象。 |
| `current_model` | `Optional[Any]` | 当前模型，通常为最近一次训练/调参/集成输出。 |
| `results` | `Optional[Any]` | 最近一次 `pull()` 得到的评分表。 |
| `models` | `Dict[str, StoredModel]` | HappyMath 内部保存的模型记录。 |

## compare

```python
compare(
    include=None,
    exclude=None,
    sort=None,
    budget_time=None,
    verbose=None,
    **kwargs,
) -> Any
```

思想：横向比较候选模型，快速找出最适合作为下一步候选的基础模型。它调用 PyCaret `compare_models()`，会训练并交叉验证候选模型，然后按指标排序。

参数：

| 参数 | 功能 |
|---|---|
| `include` | 只比较指定模型 ID 或模型对象，例如 `["lr", "dt"]`。 |
| `exclude` | 排除指定模型 ID。 |
| `sort` | 排序指标；不传时使用 `primary_metric`。 |
| `budget_time` | 训练时间预算，透传给 PyCaret。 |
| `verbose` | 是否输出 PyCaret 过程信息。 |
| `**kwargs` | 透传给 PyCaret `compare_models()`，如 `fold`、`cross_validation`、`probability_threshold`、`errors` 等。 |

输出：

- 返回排序最优的模型对象。
- 更新 `current_model` 为该模型。
- 更新 `results` 为模型比较表，通常为 `pd.DataFrame`。
- 在 `models` 中保存记录名 `compare_best`。

常见 `results` 列：

| 任务 | 常见列 |
|---|---|
| 分类 | `Model`, `Accuracy`, `AUC`, `Recall`, `Prec.`, `F1`, `Kappa`, `MCC`, `TT (Sec)` |
| 回归 | `Model`, `MAE`, `MSE`, `RMSE`, `R2`, `RMSLE`, `MAPE`, `TT (Sec)` |
| 时序 | `Model`, `MASE`, `RMSSE`, `MAE`, `RMSE`, `MAPE`, `SMAPE`, `TT (Sec)` |

注意：`compare()` 不会调参。如果比较后需要深度优化，显式调用 `tune(estimator=best, ...)`。

## create

```python
create(
    estimator,
    return_train_score=False,
    verbose=None,
    **kwargs,
) -> Any
```

思想：用指定算法训练一个模型，适合已知模型类型、需要创建多个基模型、或为后续 `tune()` / `blend()` / `stack()` 准备模型。

参数：

| 参数 | 功能 |
|---|---|
| `estimator` | PyCaret 模型 ID，如 `"lr"`、`"dt"`、`"rf"`、`"arima"`，或 sklearn 风格模型对象。 |
| `return_train_score` | 是否在评分表中包含训练集分数。 |
| `verbose` | 是否输出训练过程。 |
| `**kwargs` | 透传给 PyCaret `create_model()`，可传模型超参数或 `fold` 等。 |

输出：

- 返回训练后的模型对象。
- 更新 `results` 为交叉验证评分表。
- 若 `current_model` 为空，将当前模型设为该模型。
- 在 `models` 中保存记录名 `create_<模型名>`。

常见输出表：按折数展示指标，包含每折、`Mean` 和 `Std` 行。

## tune

```python
tune(
    estimator=None,
    n_iter=10,
    custom_grid=None,
    optimize=None,
    verbose=None,
    tuner_verbose=True,
    choose_better=True,
    **kwargs,
) -> Any
```

思想：对一个已训练模型做超参数搜索。PyCaret 官方建议先创建或比较出模型，再将模型传给 `tune_model()`；HappyMath 中对应先 `create()` / `compare()`，再 `tune()`。

参数：

| 参数 | 功能 |
|---|---|
| `estimator` | 待调参模型；不传时使用 `current_model`。 |
| `n_iter` | 搜索迭代次数，默认 `10`。越大越深，耗时越长。 |
| `custom_grid` | 自定义参数网格或搜索空间。 |
| `optimize` | 优化指标；不传时使用 `primary_metric`。 |
| `verbose` | 是否输出 PyCaret 过程信息。 |
| `tuner_verbose` | 底层调参器输出级别。脚本中常用 `False`。 |
| `choose_better` | 调参后指标变差时是否返回输入模型。默认 `True`。 |
| `**kwargs` | 透传给 PyCaret `tune_model()`，如 `search_library`、`search_algorithm`、`early_stopping`、`return_tuner`、`fold` 等。 |

输出：

- 默认返回调参后的模型对象；若 `choose_better=True` 且调参变差，可能返回输入模型。
- 如果透传 `return_tuner=True`，底层 PyCaret 可能返回 `(model, tuner)`，HappyMath 会原样接收并存储时需注意对象结构。
- 更新 `current_model` 和 `results`。
- 在 `models` 中保存记录名 `tuned`。

常见输出表：调参模型的交叉验证评分表，按 `optimize` 选择最佳参数。

## ensemble

```python
ensemble(
    estimator=None,
    method="Bagging",
    n_estimators=10,
    optimize=None,
    verbose=None,
    **kwargs,
) -> Any
```

思想：对一个基模型进行 Bagging 或 Boosting 集成，适合题目明确要求单模型集成时使用。

参数：

| 参数 | 功能 |
|---|---|
| `estimator` | 基模型；不传时使用 `current_model`。 |
| `method` | `"Bagging"` 或 `"Boosting"`，受 PyCaret 支持范围限制。 |
| `n_estimators` | 集成中基学习器数量。 |
| `optimize` | 优化指标；不传时使用 `primary_metric`。 |
| `verbose` | 是否输出过程信息。 |
| `**kwargs` | 透传给 PyCaret `ensemble_model()`。HappyMath 会过滤当前 PyCaret 版本不支持的参数。 |

输出：

- 返回集成模型，例如 `BaggingClassifier`、`BaggingRegressor` 等。
- 更新 `current_model` 和 `results`。
- 在 `models` 中保存 `ensemble_bagging` 或 `ensemble_boosting`。

## blend

```python
blend(
    estimator_list=None,
    optimize=None,
    method="auto",
    weights=None,
    verbose=None,
    **kwargs,
) -> Any
```

思想：多个模型的投票或平均融合。分类任务中通常是 VotingClassifier，回归任务中通常是 VotingRegressor 或平均器。

参数：

| 参数 | 功能 |
|---|---|
| `estimator_list` | 待融合模型列表；不传时使用 `models` 中已保存的模型。至少需要两个模型。 |
| `optimize` | 优化指标。 |
| `method` | `"auto"`、`"soft"`、`"hard"` 等。`auto` 会优先尝试 soft，不支持概率时回退。 |
| `weights` | 各模型权重。 |
| `verbose` | 是否输出过程信息。 |
| `**kwargs` | 透传给 PyCaret `blend_models()`。 |

输出：

- 返回融合模型。
- 更新 `current_model` 和 `results`。
- 在 `models` 中保存 `blended`。

注意：HappyMath 的 `compare()` 固定 `n_select=1`，不会像 PyCaret 官方示例 `compare_models(n_select=3)` 那样直接返回多个模型。需要融合时请显式 `create()` 多个模型，或手动传入 `estimator_list`。

## stack

```python
stack(
    estimator_list=None,
    meta_model=None,
    meta_model_fold=5,
    method="auto",
    restack=False,
    optimize=None,
    verbose=None,
    **kwargs,
) -> Any
```

思想：两层模型结构。第一层多个基模型输出预测，第二层元模型学习如何组合这些预测。

参数：

| 参数 | 功能 |
|---|---|
| `estimator_list` | 第一层基模型列表；不传时使用内部已保存模型。至少需要两个模型。 |
| `meta_model` | 第二层元模型；不传时使用 PyCaret 默认。 |
| `meta_model_fold` | 训练元模型时的折数。 |
| `method` | 基模型输出给元模型的方式，受 PyCaret 支持范围限制。 |
| `restack` | 是否把原始特征和基模型预测一起输入元模型。 |
| `optimize` | 优化指标。 |
| `verbose` | 是否输出过程信息。 |
| `**kwargs` | 透传给 PyCaret `stack_models()`。 |

输出：

- 返回 Stacking 模型。
- 更新 `current_model` 和 `results`。
- 在 `models` 中保存 `stacked`。

## predict

```python
predict(
    estimator=None,
    data=None,
    raw_score=False,
    verbose=None,
    **kwargs,
) -> pd.DataFrame
```

思想：用当前模型或指定模型预测 holdout/test 数据或新数据。

参数：

| 参数 | 功能 |
|---|---|
| `estimator` | 用于预测的模型；不传时使用 `current_model`。 |
| `data` | 待预测数据；不传时使用 PyCaret 可用的 holdout/test 数据。 |
| `raw_score` | 分类任务是否输出每个类别的概率分数。 |
| `verbose` | 是否输出过程信息。 |
| `**kwargs` | 透传给 PyCaret `predict_model()`，如二分类 `probability_threshold`。 |

输出：

- 返回 `pd.DataFrame`。
- 分类常见新增列：`prediction_label`、`prediction_score` 或 `prediction_score_0`、`prediction_score_1` 等。
- 回归常见新增列：`prediction_label`。
- 时序预测请使用 `TimeSeriesML.predict()`，输出通常包含 `y_pred`。

## TimeSeriesML.predict

```python
predict(
    estimator=None,
    fh=None,
    X=None,
    return_pred_int=False,
    verbose=None,
    **kwargs,
)
```

思想：对未来时间步进行预测。`fh` 表示预测步长或 forecasting horizon。

参数：

| 参数 | 功能 |
|---|---|
| `estimator` | 时序预测模型；不传时使用 `current_model`。 |
| `fh` | 预测步长；不传时使用初始化时的 `fh`。 |
| `X` | 外生变量。 |
| `return_pred_int` | 是否返回预测区间。 |
| `verbose` | 是否输出过程信息。 |
| `**kwargs` | 透传给 PyCaret time_series `predict_model()`。 |

输出：通常为 `pd.DataFrame`，至少包含 `y_pred` 列；开启预测区间时包含区间上下界列。

## finalize

```python
finalize(estimator=None) -> Any
```

思想：在全量数据上重新拟合已选模型，得到最终模型。对应 PyCaret `finalize_model()`。

参数：`estimator` 为待 finalize 的模型；不传时使用 `current_model`。

输出：

- 返回全量数据重新拟合后的模型。
- 更新 `current_model`。
- 在 `models` 中保存 `final`。

注意：`finalize()` 不会改变超参数，不会重新比较模型，也不会自动调参。

## evaluate

```python
evaluate(estimator=None) -> None
```

思想：启动 PyCaret 的交互式评估界面，主要适合 Notebook。

输出：无返回值；会调用底层 `evaluate_model()`。

## plot

```python
plot(
    estimator=None,
    plot_type="auc",
    scale=1.0,
    save=False,
    title=None,
    xlabel=None,
    ylabel=None,
    legend_title=None,
    legend_labels=None,
    figsize=(10, 6),
    plot_kwargs=None,
    font_sizes=None,
    verbose=None,
)
```

思想：调用 PyCaret `plot_model()`，并为中文标题、坐标轴、图例做兼容。`plot_type="tree"` 和 `"tree_text"` 使用 HappyMath 自定义决策树输出。

参数：

| 参数 | 功能 |
|---|---|
| `estimator` | 待绘图模型；不传时使用 `current_model`。 |
| `plot_type` | 图类型，例如分类 `auc`、`confusion_matrix`、`feature`，回归 `residuals`、`error`，聚类 `cluster`，时序 `ts` 等。 |
| `scale` | 图像缩放。 |
| `save` | 是否保存图像；PyCaret 图通常保存为文件，树文本可返回文本或写入文件。 |
| `title` / `xlabel` / `ylabel` | 标题和坐标轴文字。 |
| `legend_title` / `legend_labels` | 图例文字。 |
| `figsize` | 图像尺寸。 |
| `plot_kwargs` | 透传给底层绘图库。 |
| `font_sizes` | 中文图中各元素字号。 |
| `verbose` | 是否输出过程信息。 |

输出：

- PyCaret 图：通常返回图对象、文件路径或 `None`，取决于 PyCaret 图类型与 `save`。
- `plot_type="tree_text"`：`save=False` 时返回文本树结构；`save=True` 时返回 `.txt` 路径。
- `plot_type="tree"`：返回 Graphviz Source 或保存后的 `.png` 路径。

## scores

```python
scores(
    mode="auto",
    metrics="all",
    test_data=None,
    train_size=None,
    fold=None,
) -> pd.DataFrame
```

思想：将当前模型在不同数据切分方式下的表现整理为可直接写入报告的 DataFrame。它不训练新模型，主要用当前模型进行预测和指标计算；时序任务会复用 PyCaret 的回测结果。

参数：

| 参数 | 功能 |
|---|---|
| `mode` | 评估方式：`auto`、`holdout`、`kfold`、`leaveout`、`custom`、`train-only`。 |
| `metrics` | `"all"`、单个指标名或指标名列表；大小写不敏感。 |
| `test_data` | 自定义测试集；监督学习中必须包含目标列。 |
| `train_size` | `holdout` 模式训练集比例。 |
| `fold` | `kfold` 模式折数。 |

### scores mode 设计

| mode | 适用场景 | 支持任务 | 输出形式 |
|---|---|---|---|
| `auto` | 不确定评估方式，交给 HappyMath 根据测试集和样本量选择 | 监督、时序、聚类 | 有测试集时转 `custom`；监督小样本转 `leaveout`，中等样本转 `kfold`，大样本转 `holdout`；聚类转 `train-only`。 |
| `holdout` | 需要一次训练/测试划分对比 | 监督、时序 | index 为 `train`、`test`。 |
| `kfold` | 需要 K 折表格 | 监督、时序 | 监督任务每折一行并追加 `mean`；时序任务每个 cutoff 一行并追加 `mean`。 |
| `leaveout` | 极小样本监督学习 | 监督 | index 为 `train_mean`、`test_mean`。 |
| `custom` | 已有独立测试集 | 监督、时序 | index 为 `train`、`test`。 |
| `train-only` | 只评估训练集，或聚类内部指标 | 监督、时序、聚类 | 至少 `train` 一行；若存在测试集也包含 `test`。 |

### scores 输出列

| 任务 | 输出列特点 |
|---|---|
| 分类/回归 `holdout/custom/train-only` | 指标列名与 PyCaret 展示名一致，如 `Accuracy`、`F1`、`MAE`、`RMSE`。 |
| 分类/回归 `kfold` | 每个指标拆成 `<metric>_train` 和 `<metric>_test`，例如 `Accuracy_train`、`Accuracy_test`。 |
| 分类/回归 `leaveout` | 指标列名保持原名，行表示 LOO 平均训练/测试结果。 |
| 时序 | 指标列通常为 `MASE`、`RMSSE`、`MAE`、`RMSE`、`MAPE`、`SMAPE` 等。 |
| 聚类 | 内部指标，如 `Silhouette`。 |

异常检测当前不支持 `scores()`；建议使用 `assign()` 后结合人工标签或业务规则评价。

## get_best_model

```python
get_best_model() -> Tuple[Any, Dict[str, Any]]
```

思想：从 HappyMath 已保存模型中按 `primary_metric` 选出最佳模型。

输出：`(model, metrics)`。`model` 是模型对象，`metrics` 是从最近评分表抽取的指标字典。如果没有模型包含主指标，会返回最近保存的模型并打印提示。

## get_results

```python
get_results() -> pd.DataFrame
```

思想：读取最近一次训练、比较、调参、融合或预测后的 PyCaret 评分表。

输出：`pd.DataFrame`。如果 `results` 为空，会尝试调用底层 `pull()`；仍为空则报错。

## get_leaderboard

```python
get_leaderboard() -> pd.DataFrame
```

思想：读取 PyCaret leaderboard；若底层任务不提供，则回退到 `results` 或 `pull()`。

输出：`pd.DataFrame`，通常包含 `Model` 和指标列。

## get_metrics

```python
get_metrics() -> Any
```

思想：查看当前任务支持的指标，帮助选择 `compare(sort=...)`、`tune(optimize=...)` 和 `scores(metrics=...)`。

输出：通常为 `pd.DataFrame`，包含 `ID`、`Name` 或 `Display Name` 等列，取决于 PyCaret 版本和任务类型。

## get_models

```python
get_models() -> Iterable[str]
```

输出：HappyMath 内部已保存模型名称列表，例如 `["compare_best", "create_LogisticRegression", "tuned"]`。

## save

```python
save(model_name, model=None) -> None
```

思想：调用 PyCaret `save_model()` 保存完整 pipeline 和模型对象。

参数：`model_name` 是保存路径或文件名前缀；`model` 不传时保存 `current_model`。

输出：无 Python 返回值；底层会在磁盘写入模型文件，通常扩展名为 `.pkl`。

## load

```python
load(model_name) -> Any
```

思想：调用 PyCaret `load_model()` 加载已保存 pipeline。

输出：模型或 pipeline 对象，并不会自动设置为 `current_model`；如需作为当前模型，可手动赋值或传给 `predict(estimator=loaded, ...)`。

## get_config

```python
get_config(key=None) -> Any
```

思想：读取 PyCaret experiment 配置。适合调试数据拆分、预处理后特征、pipeline 等。

输出：

- `key=None` 时返回可用配置或配置集合，取决于 PyCaret 版本。
- 指定 `key` 时返回对应配置，例如 `get_config("X_train")`、`get_config("pipeline")`。

## 指标方向

HappyMath 内部根据指标名判断“越大越好”或“越小越好”，用于 `get_best_model()`。

| 越小越好 | 示例 |
|---|---|
| 误差 / 损失类 | `MAE`, `MSE`, `RMSE`, `RMSLE`, `MAPE`, `MASE`, `RMSSE`, `SMAPE`, `Log Loss`, `FNR`, `FPR` |

| 越大越好 | 示例 |
|---|---|
| 分类/拟合/聚类质量类 | `Accuracy`, `AUC`, `Recall`, `Prec.`, `F1`, `Kappa`, `MCC`, `R2`, `TPR`, `TNR`, `PPV`, `NPV`, `Silhouette` |

未知指标会默认按“越大越好”处理，并打印提示。因此自定义指标进入生产流程前，应确认指标方向。
