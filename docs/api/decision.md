# API 参考 - Decision

本页列出 `happymath.Decision` 模块中所有可直接使用的主要决策类。每个类都包含 `decide(...)` 入口以及结果获取方法。

## 主观赋权

::: happymath.Decision.methods.sub_weighting.SubWeighting
    options:
      members:
        - __init__
        - decide
        - get_weights
        - get_all_results
        - compare_weights
        - get_executed_methods
      show_root_heading: true
      show_root_toc_entry: false

## 客观赋权

::: happymath.Decision.methods.obj_weighting.ObjWeighting
    options:
      members:
        - __init__
        - decide
        - get_weights
        - get_all_results
        - compare_weights
        - get_executed_methods
      show_root_heading: true
      show_root_toc_entry: false

## 排序评分方法

::: happymath.Decision.methods.scoring.ScoringDecision
    options:
      members:
        - __init__
        - decide
        - get_rankings
        - get_scores
        - get_all_results
        - compare_rankings
        - compare_scores
        - get_executed_methods
      show_root_heading: true
      show_root_toc_entry: false

## 两两比较方法

::: happymath.Decision.methods.pairwise.PairwiseDecision
    options:
      members:
        - __init__
        - decide
        - get_rankings
        - get_all_results
        - compare_rankings
        - get_executed_methods
      show_root_heading: true
      show_root_toc_entry: false

## 模糊主观赋权

::: happymath.Decision.methods.fuzzy_sub_weighting.FuzzySubWeighting
    options:
      members:
        - __init__
        - decide
        - get_weights
        - get_all_results
        - compare_weights
        - get_executed_methods
      show_root_heading: true
      show_root_toc_entry: false

## 模糊客观赋权

::: happymath.Decision.methods.fuzzy_obj_weighting.FuzzyObjWeighting
    options:
      members:
        - __init__
        - decide
        - get_weights
        - get_all_results
        - compare_weights
        - get_executed_methods
      show_root_heading: true
      show_root_toc_entry: false

## 模糊排序评分

::: happymath.Decision.methods.fuzzy_scoring.FuzzyScoringDecision
    options:
      members:
        - __init__
        - decide
        - get_rankings
        - get_scores
        - get_all_results
        - compare_rankings
        - compare_scores
        - get_executed_methods
      show_root_heading: true
      show_root_toc_entry: false

## 结果管理

::: happymath.Decision.results.result_manager.ResultManager
    options:
      members:
        - get_all_results
        - get_all_weights
        - compare_weights
        - compare_rankings
        - compare_scores
      show_root_heading: true
      show_root_toc_entry: false

## 决策基类

::: happymath.Decision.core.base.DecisionBase
    options:
      members:
        - __init__
        - decide
        - get_results
        - get_executed_methods
        - clear_results
      show_root_heading: true
      show_root_toc_entry: false
