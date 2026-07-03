# Decision 模块案例演示

本案例展示 `happymath.Decision` 中主要接口的完整用法，包括：

- 客观赋权：Entropy、CRITIC
- 主观赋权：AHP
- 排序评分：TOPSIS、VIKOR
- 两两比较：ELECTRE III、PROMETHEE II
- 模糊决策：Fuzzy TOPSIS
- 结果对比：`compare_weights`、`compare_rankings`

> 以下代码已在本地 `happymath` 环境中实际运行通过。

```python
import warnings
import numpy as np

# 抑制运行过程中的方法内部警告，便于观察结果
warnings.filterwarnings('ignore')

from happymath.Decision.methods import (
    SubWeighting, ObjWeighting, ScoringDecision,
    PairwiseDecision, FuzzyScoringDecision
)

# ===========================================================================
# 决策数据：4 个备选方案 × 5 个准则
# 准则 1 为成本型(min)，其余为准效益型(max)
# ===========================================================================
decision_matrix = np.array([
    [250, 16, 12, 5, 4],
    [200, 16, 8,  3, 3],
    [300, 32, 16, 4, 4],
    [275, 32, 8,  4, 5]
], dtype=float)
criterion_type = ['min', 'max', 'max', 'max', 'max']

# ===========================================================================
# 1. 客观赋权：Entropy、CRITIC
# ===========================================================================
obj = ObjWeighting(['entropy', 'critic']).decide(
    dataset=decision_matrix,
    criterion_type=criterion_type
)
print('已执行的客观赋权方法：', obj.get_executed_methods())

# 对比两种方法得到的权重
print(obj.compare_weights())
#              entropy    critic
# Criterion_1  0.083117  0.289333
# Criterion_2  0.392194  0.224056
# Criterion_3  0.305963  0.194179
# Criterion_4  0.109362  0.147078
# Criterion_5  0.109362  0.145353

entropy_weights = obj.get_weights('entropy')
print('Entropy 权重：', entropy_weights)

# ===========================================================================
# 2. 主观赋权：AHP
# ===========================================================================
ahp_matrix = np.array([
    [1,   3,   5,   7,   9],
    [1/3, 1,   3,   5,   7],
    [1/5, 1/3, 1,   3,   5],
    [1/7, 1/5, 1/3, 1,   3],
    [1/9, 1/7, 1/5, 1/3, 1]
])
sub = SubWeighting(['ahp']).decide(dataset=ahp_matrix)
print('AHP 权重：', sub.get_weights('ahp'))
print('AHP 一致性比率 CR：', sub.get_consistency_ratios())

# ===========================================================================
# 3. 排序评分：TOPSIS、VIKOR（使用 Entropy 得到的权重）
# ===========================================================================
scoring = ScoringDecision(['topsis', 'vikor']).decide(
    dataset=decision_matrix,
    weights=entropy_weights,
    criterion_type=criterion_type
)
print('已执行的排序方法：', scoring.get_executed_methods())

# 对比不同方法的排名
print(scoring.compare_rankings())
#                topsis  vikor  Consensus
# Alternative_1     3.0    2.0          1
# Alternative_2     4.0    1.0          1
# Alternative_3     1.0    4.0          1
# Alternative_4     2.0    3.0          1

print('TOPSIS 排名：', scoring.get_rankings('topsis'))

# VIKOR 返回多组流值，get_scores 会返回完整元组
print('VIKOR 详细结果：', scoring.get_scores('vikor'))

# ===========================================================================
# 4. 两两比较：ELECTRE III、PROMETHEE II
# ===========================================================================
weights_pw = np.array([0.25, 0.15, 0.20, 0.25, 0.15])

# 各准则的阈值需与准则维度一致（长度 = 5）
Q = np.array([1, 5, 5, 2, 0.05])   # 无差异阈值
P = np.array([2, 15, 15, 5, 0.10]) # 偏好阈值
V = np.array([4, 20, 20, 8, 0.15]) # 否决阈值
S = np.array([1, 10, 10, 3, 0.08]) # 严格偏好阈值
F = np.array([5, 5, 5, 5, 5])      # 偏好函数类型

pairwise = PairwiseDecision(['electre_iii', 'promethee_ii']).decide(
    dataset=decision_matrix,
    weights=weights_pw,
    Q=Q, P=P, V=V, S=S, F=F
)
print('已执行的两两比较方法：', pairwise.get_executed_methods())
print(pairwise.compare_rankings())
#                electre_iii  promethee_ii  Consensus
# Alternative_1          1.0           1.0          1
# Alternative_2          2.0           2.0          2
# Alternative_3          3.0           3.0          3
# Alternative_4          4.0           4.0          4

print('PROMETHEE II 净流值：', pairwise.get_net_flows('promethee_ii'))

# ===========================================================================
# 5. 模糊决策：Fuzzy TOPSIS
# ===========================================================================
# 模糊决策矩阵：3 个方案 × 4 个准则，每个单元为三角模糊数 (l, m, u)
fuzzy_matrix = np.array([
    [[5, 6, 7], [3, 4, 5], [4, 5, 6], [1, 2, 3]],
    [[7, 8, 9], [6, 7, 8], [5, 6, 7], [4, 5, 6]],
    [[3, 4, 5], [4, 5, 6], [6, 7, 8], [5, 6, 7]]
], dtype=float)
fuzzy_criterion_type = ['min', 'min', 'min', 'min']

# 模糊权重：每个准则一个三角模糊数
fuzzy_weights = np.array([
    (0.30, 0.36, 0.46),
    (0.26, 0.32, 0.41),
    (0.14, 0.21, 0.29),
    (0.08, 0.10, 0.12)
])

fuzzy_scoring = FuzzyScoringDecision(['fuzzy_topsis']).decide(
    dataset=fuzzy_matrix,
    weights=fuzzy_weights,
    criterion_type=fuzzy_criterion_type
)
print('已执行的模糊排序方法：', fuzzy_scoring.get_executed_methods())
print('Fuzzy TOPSIS 排名：', fuzzy_scoring.get_rankings('fuzzy_topsis'))
print(fuzzy_scoring.compare_rankings())
```

## 关键观察

1. **统一入口**：所有类都通过 `decide(...)` 触发计算，并返回对象自身，可链式调用。
2. **自动 vs 指定方法**：构造函数传入方法名列表（如 `ObjWeighting(['entropy', 'critic'])`）可以避免运行全部可用方法，显著加快执行速度。
3. **结果对比**：`compare_weights()` 和 `compare_rankings()` 返回 `pandas.DataFrame`，便于横向比较多种方法。
4. **VIKOR 特殊输出**：`ScoringDecision.get_scores('vikor')` 返回 `(S, R, Q, solution)` 元组；如需统一排名，建议使用 `compare_rankings()`。
5. **模糊数据格式**：`dataset` 形状为 `(n_alternatives, n_criteria, 3)`，`weights` 形状为 `(n_criteria, 3)`。
