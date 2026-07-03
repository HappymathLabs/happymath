# Decision 模块快速开始

`happymath.Decision` 是 HappyMath 中用于**多准则决策（MCDM, Multi-Criteria Decision Making）**的统一框架。它将主观/客观赋权、评分排序、两两比较以及模糊决策等常用方法封装成一致的接口，方便在科研与工程问题中快速对比不同决策方法的结果。

## 主要类

| 类 | 作用 |
| --- | --- |
| `SubWeighting` | 主观赋权：AHP、BWM、FUCOM、ROC、DEMATEL 等 |
| `ObjWeighting` | 客观赋权：CRITIC、Entropy、MEREC、PSI、IDOCRIW 等 |
| `ScoringDecision` | 评分/排序：TOPSIS、VIKOR、SAW、ARAS、COPRAS、EDAS 等 |
| `PairwiseDecision` | 两两比较：ELECTRE III/IV、PROMETHEE II |
| `FuzzySubWeighting` | 模糊主观赋权：Fuzzy AHP、Fuzzy BWM、Fuzzy FUCOM |
| `FuzzyObjWeighting` | 模糊客观赋权：Fuzzy CRITIC、Fuzzy MEREC |
| `FuzzyScoringDecision` | 模糊排序评分：Fuzzy TOPSIS、Fuzzy VIKOR、Fuzzy WASPAS 等 |

## 统一使用模式

所有决策类都遵循相同的三步流程：

```python
# 1. 实例化（可传入方法名限定只运行某些方法）
obj = ObjWeighting()           # 自动选择可用方法
obj = ObjWeighting(['entropy', 'critic'])  # 只运行指定方法

# 2. 调用 decide(...) 执行计算
obj.decide(dataset=..., criterion_type=...)

# 3. 获取结果
obj.get_results()              # 原始结果字典
obj.get_weights()              # 权重（Weighting 类）
obj.get_scores()               # 得分（Scoring 类）
obj.get_rankings()             # 排名（Scoring / Pairwise / FuzzyScoring 类）
obj.get_executed_methods()     # 本次实际运行的方法列表
```

方法调用返回对象自身，因此可以链式调用：

```python
weights = ObjWeighting(['entropy']).decide(
    dataset=matrix, criterion_type=types
).get_weights()
```

## 关键参数含义

| 参数 | 含义 | 典型取值/形状 |
| --- | --- | --- |
| `dataset` / `decision_matrix` | 决策矩阵 | `np.ndarray`，形状 `(n_alternatives, n_criteria)` |
| `criterion_type` | 每个准则的方向 | 长度为 `n_criteria` 的列表，元素为 `'max'` 或 `'min'` |
| `weights` | 准则权重 | 长度为 `n_criteria` 的一维数组或列表 |
| `methods` | 构造函数参数，指定要运行的方法 | `None`（自动）、字符串或字符串列表 |

部分方法还有专属参数，例如：

- `lambda_value`：WASPAS 的组合系数（默认通常取 `0.5`）。
- `s_min`、`s_max`：SPOTIS 的各准则最小/最大满意边界。
- `P`、`Q`、`V`、`S`、`F`：ELECTRE / PROMETHEE 的偏好、无差异、否决阈值及偏好函数类型。
- `mic`、`lic`：BWM 系列方法的“最优到其它”和“其它到最差”向量。
- `criteria_rank`、`criteria_priority`：FUCOM 系列方法的准则排序与相邻优先级。

## 使用注意与已知缺陷

1. **权重未归一化会触发警告**：`DecisionBase._validate_common_parameters` 要求权重和接近 `1.0`，否则会在运行时发出 `UserWarning`，内部通常会自动归一化。
2. **自动选择方法可能运行很多方法**：不指定 `methods` 时，框架会根据输入参数把所有“参数够用”的方法都运行一遍。模糊方法或迭代型方法（如 Fuzzy TOPSIS、Fuzzy MEREC）在数据量稍大时会显著变慢。
3. **部分方法对输入维度敏感**：例如 `spotis` 必须提供 `s_min`、`s_max`；`smart` 必须提供 `grades`、`lower`、`upper` 和 `utility_functions`；缺少对应参数时这些方法会被跳过。
4. **Pairwise 类阈值需要与准则维度对齐**：`P`、`Q`、`V`、`S`、`F` 的长度必须等于准则数。
5. **模糊矩阵格式**：模糊方法的 `dataset` 通常是三维数组 `(alternatives, criteria, 3)`，每个元素为三角模糊数 `(l, m, u)`；权重也要求是 `(criteria, 3)` 或外层再包一层 `[[(l, m, u), ...]]`。
6. **错误以警告形式跳过**：`decide()` 内部对每个方法都包了 `try/except`，失败时只打印 `UserWarning` 而不中断整个流程，因此调用后应检查 `get_executed_methods()` 确认目标方法是否真的成功。

## 最小可运行示例：Entropy + TOPSIS

下面的例子展示了如何用客观赋权（Entropy）得到权重，再用 TOPSIS 对方案进行排序。

```python
import numpy as np
from happymath.Decision.methods.obj_weighting import ObjWeighting
from happymath.Decision.methods.scoring import ScoringDecision

# 4 个备选方案 × 3 个准则
# 准则 1 为成本型(min)，准则 2、3 为效益型(max)
dm = np.array([
    [250, 16, 12],
    [200, 16, 8],
    [300, 32, 16],
    [275, 32, 8]
])
criterion_type = ['min', 'max', 'max']

# 1) 用 Entropy 计算客观权重（指定 method 获取单个权重向量）
weights = ObjWeighting(['entropy']).decide(
    dataset=dm,
    criterion_type=criterion_type
).get_weights(method='entropy')
print('Entropy weights:', weights)

# 2) 用 TOPSIS 排序
ranking = ScoringDecision(['topsis']).decide(
    dataset=dm,
    weights=weights,
    criterion_type=criterion_type
).get_rankings()
print('TOPSIS ranking:', ranking)   # 数值越小表示排名越靠前
```

运行后将输出每个准则的 Entropy 权重以及各方案的 TOPSIS 排名（`1` 为最优）。
