# Decision 模块 API 文档

本页按类组织 `happymath.Decision` 模块的主要公共接口。每个类都继承自 `DecisionBase`，因此共享 `decide(...)`、`get_results()`、`get_executed_methods()` 等统一入口。

---

## 目录

- [DecisionBase](#decisionbase)
- [SubWeighting](#subweighting)
- [ObjWeighting](#objweighting)
- [ScoringDecision](#scoringdecision)
- [PairwiseDecision](#pairwisedecision)
- [FuzzySubWeighting](#fuzzysubweighting)
- [FuzzyObjWeighting](#fuzzyobjweighting)
- [FuzzyScoringDecision](#fuzzyscoringdecision)
- [ResultManager](#resultmanager)

---

## DecisionBase

所有决策类的抽象基类，提供统一生命周期、参数校验与结果管理。

### `__init__(methods=None)`

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `methods` | `str` / `List[str]` / `None` | 指定要运行的方法；为 `None` 时根据输入参数自动选择。 |

### `decide(**kwargs) -> DecisionBase`

通用执行入口。

- 调用 `_validate_common_parameters` 校验 `dataset`/`decision_matrix`、`weights`、`criterion_type` 等。
- 若子类实现了 `_validate_inputs`，则进一步校验并标准化参数。
- 根据 `methods` 自动筛选或严格校验方法。
- 逐个执行方法，失败时发出 `UserWarning` 并跳过。
- 返回 `self`，支持链式调用。

### 公共结果访问方法

| 方法 | 返回 | 说明 |
| --- | --- | --- |
| `get_results()` | `Dict[str, Any]` | 返回所有原始结果字典。 |
| `get_result(method)` | `Any` / `None` | 返回指定方法的原始结果。 |
| `get_executed_methods()` | `List[str]` | 返回本次实际执行的方法名列表。 |
| `clear_results()` | `DecisionBase` | 清空结果与状态，返回 `self`。 |

### 通用参数校验规则

- `dataset` / `decision_matrix`：必须是二维非空 `numpy.ndarray` 或列表。
- `weights`：一维数组，不能为负；若求和不为 `1.0` 会触发警告。
- `criterion_type`：元素只能是 `'max'` 或 `'min'` 的列表。

---

## SubWeighting

主观赋权方法集合，基于专家判断计算准则权重。

继承：`DecisionBase`

### `__init__(methods=None)`

含义与基类相同。

### 支持的方法

| 方法名 | 必需参数 | 说明 |
| --- | --- | --- |
| `ahp` | `dataset` | 层次分析法，输入成对比较矩阵。 |
| `bwm` | `mic`, `lic` | Best-Worst Method，最优/最差比较向量。 |
| `simplified_bwm` | `mic`, `lic` | 简化 BWM。 |
| `fucom` | `criteria_rank`, `criteria_priority` | Full Consistency Method。 |
| `roc` | `criteria_rank` | Rank Order Centroid。 |
| `rrw` | `criteria_rank` | Rank Reciprocal Weighting。 |
| `rsw` | `criteria_rank` | Rank Sum Weights。 |
| `dematel` | `dataset` | 直接影响关系矩阵。 |
| `wings` | `dataset` | Weighted Influence Non-linear Gauge System。 |

### `get_weights(method=None)`

- `method=None`：返回 `{方法名: 权重数组}` 字典。
- `method='ahp'` 等：返回单个权重向量。
- 对 `ahp`、`simplified_bwm` 会提取元组第一个元素；对 `dematel`、`wings` 提取第三个元素。

### `get_consistency_ratios()`

返回 `{'ahp': CR, 'simplified_bwm': CR}` 形式的一致性比率字典（仅当对应方法成功执行）。

### `compare_weights(methods=None)`

返回 `DataFrame`，列是方法名，行是准则，便于横向比较权重。

### `get_all_results()`

返回 `ResultManager` 中所有处理后的结果字典。

---

## ObjWeighting

客观赋权方法集合，从决策矩阵的数据结构中提取准则权重。

继承：`DecisionBase`

### `__init__(methods=None)`

含义与基类相同。

### 支持的方法

| 方法名 | 必需参数 | 说明 |
| --- | --- | --- |
| `critic` | `dataset`, `criterion_type` | 基于指标相关性与对比度。 |
| `entropy` | `dataset`, `criterion_type` | 熵权法。 |
| `idocriw` | `dataset`, `criterion_type` | Integrated Determination of Objective CRIteria Weights。 |
| `merec` | `dataset`, `criterion_type` | 基于指标剔除效应。 |
| `mpsi` | `dataset`, `criterion_type` | Modified Preference Selection Index。 |
| `seca` | `dataset`, `criterion_type` | Simultaneous Evaluation of Criteria and Alternatives。 |
| `cilos` | `dataset`, `criterion_type` | Criterion Impact LOSs。 |

### `get_weights(method=None)`

- `method=None`：返回所有方法权重的字典。
- 指定方法名时返回单个权重向量；若结果是元组，取第一个元素。

### `compare_weights(methods=None, add_stats=False)`

- `methods`：要对比的方法列表，`None` 表示全部。
- `add_stats=True`：额外附加 `Mean`、`Std`、`Min`、`Max` 统计列。

### `get_all_results()`

返回所有处理后的结果。

---

## ScoringDecision

评分/排序方法集合，综合决策矩阵、权重与准则方向给出方案得分和排名。

继承：`DecisionBase`

### `__init__(methods=None)`

含义与基类相同。

### 常用方法

| 方法名 | 必需参数 | 说明 |
| --- | --- | --- |
| `topsis` | `dataset`, `criterion_type` | 逼近理想解排序法。 |
| `vikor` | `dataset`, `criterion_type` | 多准则妥协解。 |
| `saw` | `dataset`, `criterion_type` | 简单加权求和。 |
| `aras` | `dataset`, `criterion_type` | Additive Ratio ASsessment。 |
| `copras` | `dataset`, `criterion_type` | COmplex PRoportional ASsessment。 |
| `edas` | `dataset`, `criterion_type` | 基于平均解距离。 |
| `codas` | `dataset`, `criterion_type` | 组合距离评估。 |
| `cocoso` | `dataset`, `criterion_type` | Combined Compromise Solution。 |
| `mabac` | `dataset`, `criterion_type` | 多属性边界逼近。 |
| `mairca` | `dataset`, `criterion_type` | 理想-现实比较分析。 |
| `marcos` | `dataset`, `criterion_type` | Measurement of Alternatives and Ranking。 |
| `moora` | `dataset`, `criterion_type` | 多目标比率分析。 |
| `moosra` | `dataset`, `criterion_type` | 简化 MOORA。 |
| `multimoora` | `dataset`, `criterion_type` | MOORA + Full Multiplicative Form。 |
| `waspas` | `dataset`, `criterion_type`, `lambda_value` | 加权求和与乘积组合。 |
| `wisp` | `dataset`, `criterion_type` | Weighted Sum-Product。 |
| `todim` | `dataset`, `criterion_type` | 交互式多准则决策。 |
| `gra` | `dataset`, `criterion_type` | 灰色关联分析。 |
| `lmaw` | `dataset`, `criterion_type` | 对数加法权重法。 |
| `rafsi` | `dataset`, `criterion_type` | 函数映射区间排序。 |
| `spotis` | `dataset`, `criterion_type`, `s_min`, `s_max` | Stable Preference Ordering。 |
| `smart` | `dataset`, `criterion_type`, `grades`, `lower`, `upper` | 简单多属性评级。 |
| `maut` | `dataset`, `criterion_type`, `utility_functions` | 多属性效用理论。 |
| `psi` | `dataset`, `criterion_type` | Preference Selection Index。 |
| `borda` | `dataset`, `criterion_type` | Borda 计数。 |
| `copeland` | `dataset`, `criterion_type` | Copeland 法。 |
| 等 | — | 还包括 `macbeth`、`mara`、`ocra`、`oreste`、`piv`、`rov` 等。 |

注意：

- 多数方法默认 `weights` 为等权重或方法内部权重；`weights` 为可选参数。
- `waspas` 必须提供 `lambda_value`。
- `spotis` 必须提供 `s_min`、`s_max`。
- `smart` 必须提供 `grades`、`lower`、`upper`。
- `maut` 必须提供 `utility_functions`。

### `get_scores(method=None)`

- `method=None`：返回所有方法得分的字典。
- 若结果为标准二维数组 `[id, score]`，提取第二列作为得分。
- `vikor` 返回完整元组 `(S, R, Q, solution)`。

### `get_rankings(method=None)`

- 对一维得分使用 `np.argsort(-scores) + 1` 计算排名（分数越高排名越靠前）。
- `method=None` 返回所有方法排名的字典。

### `compare_scores(methods=None)` / `compare_rankings(methods=None)`

返回 `DataFrame`，列是方法名，行是方案名；`compare_rankings` 在多方法时附加 `Consensus` 共识排名。

---

## PairwiseDecision

两两比较/优序方法集合，包含 ELECTRE 与 PROMETHEE 家族。

继承：`DecisionBase`

### `__init__(methods=None)`

含义与基类相同。

### 支持的方法

| 方法名 | 必需参数 | 说明 |
| --- | --- | --- |
| `electre_iii` | `dataset`, `weights`, `P`, `Q`, `V` | ELECTRE III，使用一致性、反一致性与可信度。 |
| `electre_iv` | `dataset`, `P`, `Q`, `V` | ELECTRE IV，不使用权重。 |
| `promethee_ii` | `dataset`, `weights`, `Q`, `S`, `P`, `F` | PROMETHEE II，输出净流值与排名。 |

### 参数说明

| 参数 | 说明 |
| --- | --- |
| `weights` | 准则权重，长度等于准则数。 |
| `P` | 偏好阈值（preference threshold）。 |
| `Q` | 无差异阈值（indifference threshold）。 |
| `V` | 否决阈值（veto threshold）。 |
| `S` | 严格偏好阈值（strict preference threshold）。 |
| `F` | 偏好函数类型数组。 |

以上阈值数组长度均需与准则数一致。

### 结果访问方法

| 方法 | 返回 | 说明 |
| --- | --- | --- |
| `get_kernel(method)` | `set` / `None` | 获取 ELECTRE 的核集（非支配解）。 |
| `get_outranking_matrix(method)` | `np.ndarray` / `None` | 获取优势/支配矩阵。 |
| `get_net_flows(method='promethee_ii')` | `np.ndarray` / `None` | 获取 PROMETHEE II 的净流值。 |
| `compare_rankings(methods=None)` | `DataFrame` | 对比各方法的排名。 |

---

## FuzzySubWeighting

模糊主观赋权方法集合。

继承：`DecisionBase`

### `__init__(methods=None)`

含义与基类相同。

### 支持的方法

| 方法名 | 必需参数 | 说明 |
| --- | --- | --- |
| `fuzzy_ahp` | `dataset` | 模糊成对比较矩阵，形状 `(n_criteria, n_criteria, 3)`。 |
| `fuzzy_bwm` | `mic`, `lic` | 模糊 Best-Worst Method，向量为三角模糊数列表。 |
| `fuzzy_fucom` | `criteria_rank`, `criteria_priority` | 模糊 FUCOM。 |

### 结果访问方法

| 方法 | 说明 |
| --- | --- |
| `get_fuzzy_weights(method=None)` | 获取三角模糊权重；`fuzzy_ahp` 取元组第 1 个，`fuzzy_bwm` 取第 3 个。 |
| `get_defuzzified_weights(method=None)` | 获取去模糊化后的清晰权重。 |
| `get_weights(method=None)` | 等价于 `get_defuzzified_weights`。 |
| `get_consistency_ratios()` | 返回 `{'fuzzy_ahp': CR, 'fuzzy_bwm': CR}`。 |
| `compare_weights(methods=None)` | 返回 `DataFrame`。 |

---

## FuzzyObjWeighting

模糊客观赋权方法集合。

继承：`DecisionBase`

### `__init__(methods=None)`

含义与基类相同。

### 支持的方法

| 方法名 | 必需参数 | 说明 |
| --- | --- | --- |
| `fuzzy_critic` | `dataset`, `criterion_type` | 模糊 CRITIC。 |
| `fuzzy_merec` | `dataset`, `criterion_type` | 模糊 MEREC。 |

### `decide(dataset, criterion_type, **kwargs)`

模糊矩阵 `dataset` 形状通常为 `(n_alternatives, n_criteria, 3)`。

### 结果访问方法

| 方法 | 说明 |
| --- | --- |
| `get_fuzzy_weights(method=None)` | 返回模糊权重结果（原始）。 |
| `get_weights(method=None)` | 返回去模糊化后的清晰权重；元组结果取第一个元素。 |
| `compare_weights(methods=None, add_stats=False)` | 返回 `DataFrame`。 |

---

## FuzzyScoringDecision

模糊评分/排序方法集合。

继承：`DecisionBase`

### `__init__(methods=None)`

含义与基类相同。

### 支持的方法

| 方法名 | 必需参数 | 说明 |
| --- | --- | --- |
| `fuzzy_aras` | `dataset`, `weights`, `criterion_type` | 模糊 ARAS。 |
| `fuzzy_copras` | `dataset`, `weights`, `criterion_type` | 模糊 COPRAS。 |
| `fuzzy_edas` | `dataset`, `weights`, `criterion_type` | 模糊 EDAS。 |
| `fuzzy_moora` | `dataset`, `weights`, `criterion_type` | 模糊 MOORA。 |
| `fuzzy_ocra` | `dataset`, `weights`, `criterion_type` | 模糊 OCRA。 |
| `fuzzy_topsis` | `dataset`, `weights`, `criterion_type` | 模糊 TOPSIS。 |
| `fuzzy_vikor` | `dataset`, `weights`, `criterion_type` | 模糊 VIKOR。 |
| `fuzzy_waspas` | `dataset`, `weights`, `criterion_type` | 模糊 WASPAS。 |

### 数据格式

- `dataset`：三维数组 `(n_alternatives, n_criteria, 3)`，每个元素为 `(l, m, u)`。
- `weights`：二维数组 `[[(l, m, u), ...]]` 或 `(n_criteria, 3)`。

### 结果访问方法

| 方法 | 说明 |
| --- | --- |
| `get_scores(method=None)` | 获取模糊得分。 |
| `get_rankings(method=None)` | 获取排名；对一维得分使用高分优先。 |
| `compare_scores(methods=None)` | 返回 `DataFrame`。 |
| `compare_rankings(methods=None)` | 返回 `DataFrame`，多方法时附加 `Consensus`。 |

---

## ResultManager

多方法结果的统一存储、处理与比较器。

### 主要方法

| 方法 | 说明 |
| --- | --- |
| `add_result(method_name, raw_output, metadata=None)` | 添加单个原始结果。 |
| `get_result(method_name)` | 获取 `MethodResult` 对象。 |
| `get_all_results()` | 返回 `{方法名: processed_data}` 字典。 |
| `get_all_weights()` | 返回所有可提取权重的字典。 |
| `get_all_scores()` | 返回所有可提取得分的字典。 |
| `get_all_rankings()` | 返回所有可提取排名的字典。 |
| `get_weights(method_name=None)` | 获取指定方法权重或全部权重。 |
| `get_rankings(method_name=None)` | 获取指定方法排名或全部排名。 |
| `compare_weights(methods=None, add_stats=False)` | 权重对比 `DataFrame`。 |
| `compare_scores(methods=None)` | 得分对比 `DataFrame`。 |
| `compare_rankings(methods=None)` | 排名对比 `DataFrame`，含 `Consensus`。 |
| `compare_classifications(methods=None)` | 分类结果对比 `DataFrame`。 |

### 输出处理规则（MethodResult）

`MethodResult` 在构造时会自动解析原始输出：

- `ahp`、`simplified_bwm`：分离 `weights` 与 `consistency_ratio`。
- `dematel`、`wings`：分离 `prominence`、`relation`、`weights`。
- `vikor` / `fuzzy_vikor`：分离 `flow_s`、`flow_r`、`flow_q`、`solution`。
- `waspas` / `fuzzy_waspas`：分离 `wsm`、`wpm`、`waspas`。
- `multimoora`：分离 `flow_1`、`flow_2`、`flow_3`、`flow_final`。
- `electre_iii` / `electre_iv`：提取 `credibility`、`rank_d`、`rank_a`、`rank_m`、`rank_p`。
- `promethee_ii`：提取 `flow` 与排名。
- 标准评分方法：默认二维数组 `[id, score]`，自动提取 `scores` 与 `ranking`。

---

## 通用注意事项

1. **自动选择会运行所有可用方法**：如果不限制 `methods`，模糊或迭代方法可能很慢。
2. **失败只跳过，不中断**：`decide()` 内部对每个方法单独捕获异常，调用后应检查 `get_executed_methods()`。
3. **权重归一化警告**：`weights` 求和不等于 `1.0` 时会发出 `UserWarning`。
4. **维度敏感**：模糊矩阵、阈值数组、权重向量的长度/维度必须与准则数一致。
