# API 参考 - AutoML

本页列出 `happymath.AutoML` 模块中可供用户直接使用的主要类。每个类都包含完整的方法签名与文档说明。

## 子模块导览

::: happymath.AutoML
    options:
      members: false
      show_submodules: true
      show_root_heading: true
      show_root_toc_entry: false

## AutoML 基类

::: happymath.AutoML.base.AutoMLBase
    options:
      members:
        - __init__
        - compare
        - create
        - tune
        - ensemble
        - blend
        - stack
        - predict
        - finalize
        - evaluate
        - plot
        - scores
        - get_best_model
        - get_results
        - get_leaderboard
        - get_metrics
        - get_models
        - save
        - load
        - get_config
      show_root_heading: true
      show_root_toc_entry: false

## 分类任务

::: happymath.AutoML.supervised.ClassificationML
    options:
      members:
        - __init__
      show_root_heading: true
      show_root_toc_entry: false

## 回归任务

::: happymath.AutoML.supervised.RegressionML
    options:
      members:
        - __init__
      show_root_heading: true
      show_root_toc_entry: false

## 聚类任务

::: happymath.AutoML.unsupervised.ClusteringML
    options:
      members:
        - __init__
        - create
        - assign
      show_root_heading: true
      show_root_toc_entry: false

## 异常检测任务

::: happymath.AutoML.unsupervised.AnomalyML
    options:
      members:
        - __init__
        - create
        - assign
      show_root_heading: true
      show_root_toc_entry: false

## 时序预测任务

::: happymath.AutoML.time_series.TimeSeriesML
    options:
      members:
        - __init__
        - predict
      show_root_heading: true
      show_root_toc_entry: false
