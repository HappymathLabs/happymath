# Decision 多准则决策分析

`happymath.Decision` 模块提供了一套完整的多准则决策分析（MCDM）框架，包含主观赋权、客观赋权、排序方法、两两比较方法以及对应的模糊决策方法。

## 模块结构

| 类名 | 说明 | 典型算法 |
|------|------|----------|
| `SubWeighting` | 主观赋权 | AHP、BWM、FUCOM、ROC 等 |
| `ObjWeighting` | 客观赋权 | CRITIC、Entropy、MEREC、PSI、SECA、CILOS 等 |
| `ScoringDecision` | 排序评分 | TOPSIS、VIKOR、SAW、MOORA、WASPAS 等 |
| `PairwiseDecision` | 两两比较 | ELECTRE、PROMETHEE 家族等 |
| `FuzzySubWeighting` | 模糊主观赋权 | 模糊 AHP、模糊 BWM 等 |
| `FuzzyObjWeighting` | 模糊客观赋权 | 模糊 CRITIC、模糊 Entropy 等 |
| `FuzzyScoringDecision` | 模糊排序 | 模糊 TOPSIS、模糊 VIKOR 等 |

## 统一使用模式

所有决策类都遵循相同的使用模式：

1. 实例化决策类（可指定 `methods` 参数选择特定算法）
2. 调用 `.decide(...)` 传入数据与参数
3. 使用 `.get_results()`、`.get_rankings()`、`.get_weights()` 等方法获取结果

## 客观赋权 + TOPSIS 排序示例

```python
from happymath.Decision import ObjWeighting, ScoringDecision
import numpy as np

# 决策矩阵：3 个方案，3 个指标
dm_data = np.array([[250, 16, 12], [200, 16, 8], [300, 32, 16]])
criteria = ["min", "max", "max"]

# 使用熵权法计算客观权重
weighting = ObjWeighting(methods=["entropy"])
weights = weighting.decide(
    dataset=dm_data, criterion_type=criteria
).get_weights(method="entropy")
print("Entropy 权重:", weights)

# 使用 TOPSIS 排序
scoring = ScoringDecision(methods=["topsis"])
rankings = scoring.decide(
    dataset=dm_data, weights=weights, criterion_type=criteria
).get_rankings(method="topsis")
print("TOPSIS 排序:", rankings)
```

## 多方法对比示例

```python
from happymath.Decision import ObjWeighting
import numpy as np

dm_data = np.array([[10, 20, 5, 80], [12, 18, 6, 85], [11, 22, 5.5, 78]])
criteria = ["min", "max", "min", "max"]

# 不指定 methods，自动运行所有可用方法
weighting = ObjWeighting()
weighting.decide(dataset=dm_data, criterion_type=criteria)

print("执行的方法:", weighting.get_executed_methods())
print("所有权重:")
print(weighting.compare_weights())
```

## 主观赋权示例（AHP）

```python
from happymath.Decision import SubWeighting
import numpy as np

ahp_matrix = np.array([
    [1, 3, 5, 7],
    [1/3, 1, 3, 5],
    [1/5, 1/3, 1, 3],
    [1/7, 1/5, 1/3, 1]
])

sub = SubWeighting(methods=["ahp"])
weights = sub.decide(dataset=ahp_matrix).get_weights(method="ahp")
print("AHP 权重:", weights)
```

## 两两比较示例（ELECTRE/PROMETHEE）

```python
from happymath.Decision import PairwiseDecision
import numpy as np

dm_data = np.array([
    [5, 85, 70, 15, 0.8],
    [4, 92, 65, 12, 0.9],
    [5, 80, 85, 20, 0.7],
    [6, 75, 80, 18, 0.85]
])
weights = np.array([0.25, 0.30, 0.20, 0.15, 0.10])
thresholds = {
    "Q": np.array([1, 5, 5, 2, 0.05]),
    "P": np.array([2, 15, 15, 5, 0.1]),
    "V": np.array([4, 20, 20, 8, 0.15]),
    "S": np.array([1, 10, 10, 3, 0.08]),
    "F": np.array([5, 5, 5, 5, 5]),
    "B": np.array([3, 70, 60, 10, 0.6])
}

pairwise = PairwiseDecision()
pairwise.decide(dataset=dm_data, weights=weights, **thresholds)
print("执行的方法:", pairwise.get_executed_methods())
print(pairwise.compare_rankings())
```

## 模糊决策示例

```python
from happymath.Decision import FuzzyScoringDecision
import numpy as np

# 三角模糊决策矩阵
fuzzy_dm = np.array([
    [[5, 6, 7], [3, 4, 5], [4, 5, 6], [1, 2, 3]],
    [[7, 8, 9], [6, 7, 8], [5, 6, 7], [4, 5, 6]],
    [[3, 4, 5], [4, 5, 6], [6, 7, 8], [5, 6, 7]]
])
fuzzy_weights = np.array([
    (0.30, 0.36, 0.46),
    (0.26, 0.32, 0.41),
    (0.14, 0.21, 0.29),
    (0.08, 0.10, 0.12)
])
criteria = ["min", "min", "min", "min"]

fuzzy_scoring = FuzzyScoringDecision()
rankings = fuzzy_scoring.decide(
    dataset=fuzzy_dm, weights=fuzzy_weights, criterion_type=criteria
).get_rankings(method="fuzzy_topsis")
print("模糊 TOPSIS 排序:", rankings)
```

## 常用方法速查

- `decide(...)`：执行决策方法
- `get_executed_methods()`：获取已执行的方法列表
- `get_results()`：获取所有原始结果
- `get_weights(method)`：获取指定方法的权重
- `get_rankings(method)`：获取指定方法的排序
- `get_scores(method)`：获取指定方法的评分
- `compare_weights()` / `compare_rankings()` / `compare_scores()`：跨方法对比

## 参数说明

- `dataset` / `decision_matrix`：决策矩阵，形状为 `(n_alternatives, n_criteria)`
- `criterion_type`：指标类型列表，每个元素为 `"min"` 或 `"max"`
- `weights`：权重向量，长度为 `n_criteria`
- `methods`：算法名称或列表，不指定时自动选择
