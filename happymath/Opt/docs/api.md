# Opt 模块 API 文档

本页按类/函数组织，说明 `happymath.Opt` 中常用的建模、求解与结果获取接口。

---

## 优化主入口

### `happymath.Opt.OptModule.OptModule`

统一的优化问题入口类，继承自 `OptBase`。

```python
OptModule(
    obj_func,
    constraints=None,
    mode="auto",
    default_search_range=100,
    show_bound_warnings=True,
    tighten_bounds=None,
    **kwargs
)
```

#### 参数

| 参数 | 类型 | 说明 | 默认值 |
| --- | --- | --- | --- |
| `obj_func` | `dict` | 目标函数字典，如 `{"min": expr}`、`{"max": expr}`，多目标可传 `{"min": [expr1, expr2]}` | 必填 |
| `constraints` | `list` / `None` | 约束列表，元素为 SymPy 关系表达式 | `None` |
| `mode` | `str` | 求解库模式：`"auto"`、`"pyomo"`、`"pymoo"` | `"auto"` |
| `default_search_range` | `float` / `int` | 变量未显式给出边界时的默认搜索半径 | `100` |
| `show_bound_warnings` | `bool` | 是否在变量边界不完整时发出警告 | `True` |
| `tighten_bounds` | `str` / `None` | 边界紧化策略，如 `"none"`、`"rbc"`、`"lp"` | `None` |
| `**kwargs` | — | 额外参数，可传入 `functional_config` / `functional_ode` 等功能型配置 | — |

#### 主要属性

| 属性 | 说明 |
| --- | --- |
| `mode` | 当前求解库模式 |
| `libraries` | 实际启用的后端列表，如 `["pyomo", "pymoo"]` 或 `["pymoo"]` |
| `is_single_obj` | 是否为单目标问题 |
| `parse_result` | 解析结果 `ParseResult` 对象 |
| `pyomo_solver` | 内部 `PyomoSolver` 实例 |
| `pymoo_solver` | 内部 `PymooSolver` 实例 |

#### 方法

##### `solve(solver=None, use_auto_solvers=True, max_solvers=3, ref=None)`

求解优化问题。

| 参数 | 类型 | 说明 | 默认值 |
| --- | --- | --- | --- |
| `solver` | `str` / `list` / `None` | 求解器或算法名；传 `None` 自动选择 | `None` |
| `use_auto_solvers` | `bool` | 是否自动尝试多个求解器 | `True` |
| `max_solvers` | `int` | 最大尝试求解器数量 | `3` |
| `ref` | `dict` / `None` | Pymoo 多目标后处理参考点 | `None` |

返回值：`OptResult`。

---

## 优化结果容器

### `happymath.Opt.results.opt_result.OptResult`

统一包装 PyomoSolver 与 PymooSolver 的返回结果。

```python
OptResult(results, opt_module_info)
```

#### 参数

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `results` | `dict` / `list[dict]` | 求解器返回的单个结果或结果列表 |
| `opt_module_info` | `dict` | OptModule 的基本信息字典 |

#### 属性

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `success` | `bool` | 是否存在成功的求解结果 |
| `variables` | `dict[str, float]` | 最优解的变量取值 |
| `objective_value` | `float` / `list[float]` / `None` | 最优解的目标值；单目标为标量，多目标为列表 |
| `solution` | `dict` / `None` | 包含 `variables` 与 `objective_value` 的字典 |
| `all_solutions` | `list[dict]` | 所有成功解的列表（带缓存） |
| `all_solvers` | `list[dict]` | 所有求解器运行信息（带缓存） |
| `raw_solution` | `dict` / `None` | 原始最优结果字典 |
| `raw_all_solutions` | `list[dict]` | 所有原始结果字典的副本 |
| `solver` | `dict` / `None` | 最优结果对应的求解器信息 |
| `message` | `str` | 求解消息 |
| `solver_name` | `str` | 求解器名称（兼容旧接口） |
| `pareto_front` | `list[dict]` | 多目标问题的 Pareto 前沿；每个元素含 `objectives`、`variables_array`、`algorithm` |

#### 方法

| 方法 | 说明 |
| --- | --- |
| `clear_cache()` | 清除 `all_solutions`、`all_solvers`、`pareto_front` 的缓存 |

---

## 优化基类

### `happymath.Opt.opt_core.opt_base.OptBase`

轻量级协调器，负责调用 `ExpressionProcessor` 并缓存解析结果。

```python
OptBase(
    obj_func,
    constraints=None,
    epsilon=1e-6,
    default_search_range=100,
    show_bound_warnings=True,
    tighten_bounds=None,
    pyomo_config=None,
    pymoo_config=None,
    **kwargs
)
```

#### 参数

| 参数 | 说明 |
| --- | --- |
| `obj_func` | 目标函数字典 |
| `constraints` | 约束列表 |
| `epsilon` | 严格不等式处理时使用的 epsilon |
| `default_search_range` | 默认搜索半径 |
| `show_bound_warnings` | 是否显示边界警告 |
| `tighten_bounds` | 边界紧化策略 |
| `pyomo_config` / `pymoo_config` | 预留的求解器配置 |
| `**kwargs` | 额外参数，可传入功能型配置 |

#### 属性

| 属性 | 说明 |
| --- | --- |
| `parse_result` | 返回 `ParseResult` 解析结果对象 |
| `pyomo_problem_type` | 返回 Pyomo 问题类型字符串，如 `"LP"`、`"QP"`、`"NLP"`、`"MILP"` 等 |
| `pymoo_problem_type` | 返回 Pymoo 问题类型字典 |
| `epsilon` | 当前 epsilon 值 |
| `default_search_range` | 当前默认搜索半径 |

#### 方法

| 方法 | 说明 |
| --- | --- |
| `clear_cache()` | 清除 Pyomo 模型与 Pymoo 问题缓存 |

---

## Pyomo 求解器

### `happymath.Opt.solvers.pyomo_solver.PyomoSolver`

处理所有 Pyomo 相关的求解逻辑，继承自 `BaseSolver`。

```python
PyomoSolver(problem: IProblemDefinition, epsilon: float = 1e-6)
```

#### 参数

| 参数 | 说明 |
| --- | --- |
| `problem` | 问题定义接口，通常为 `ParseResult` |
| `epsilon` | 约束处理 epsilon |

#### 方法

##### `solve(solver=None, use_auto_solvers=True, max_solvers=3)`

统一求解方法，返回 `list[dict]`。

| 参数 | 说明 |
| --- | --- |
| `solver` | 求解器名称或名称列表；`None` 时按问题类型自动选择 |
| `use_auto_solvers` | 是否自动尝试多个求解器 |
| `max_solvers` | 最大尝试数量，传 `"all"` 表示尝试全部候选 |

自动候选策略：

- LP：`cbc`、`glpk`
- QP：`scip`、`ipopt`
- NLP：`scip`、`ipopt`
- MILP：`scip`、`cbc`
- MIQP / MINLP：`scip`、`mindtpy`

##### `get_available_solvers()`

返回当前环境中可用的 Pyomo 求解器名称列表。

---

## Pymoo 求解器

### `happymath.Opt.solvers.pymoo_solver.PymooSolver`

处理所有 Pymoo 进化算法相关的求解逻辑，继承自 `BaseSolver`。

```python
PymooSolver(problem: IProblemDefinition, epsilon: float = 1e-6)
```

#### 参数

| 参数 | 说明 |
| --- | --- |
| `problem` | 问题定义接口，通常为 `ParseResult` |
| `epsilon` | 约束处理 epsilon |

#### 属性

| 属性 | 说明 |
| --- | --- |
| `algorithm_factory` | `PymooAlgorithmFactory` 实例，负责算法推荐与创建 |
| `_budget_override` | 可手动覆盖的评估预算上限；设为 `int` 可降低运行时间 |

#### 方法

##### `solve(solver=None, use_auto_solvers=True, max_solvers=3)`

统一求解方法，返回 `list[dict]`。

| 参数 | 说明 |
| --- | --- |
| `solver` | 算法名或列表；`None` 时由 `algorithm_factory` 推荐 |
| `use_auto_solvers` | 是否自动尝试多个算法 |
| `max_solvers` | 最大尝试数量 |

##### `get_available_solvers()`

返回 `algorithm_factory` 中所有可用算法名称列表。

---

## 表达式处理器

### `happymath.Opt.opt_expr.processor.ExpressionProcessor`

统一的表达式处理入口，协调目标/约束分析器、变量管理器、边界管理器与问题类型分析器。

#### 方法

##### `process(obj_func, constraints=None, default_search_range=100, epsilon=1e-6, show_bound_warnings=True, tighten_bounds=None, **kwargs)`

处理优化问题的表达式并返回 `ParseResult`。

| 参数 | 说明 |
| --- | --- |
| `obj_func` | 目标函数字典 |
| `constraints` | 约束列表 |
| `default_search_range` | 默认搜索半径 |
| `epsilon` | 严格不等式 epsilon |
| `show_bound_warnings` | 是否显示边界警告 |
| `tighten_bounds` | 边界紧化策略 |
| `**kwargs` | 额外选项，可包含 `functional_config` / `functional_ode` |

---

## 解析结果

### `happymath.Opt.opt_expr.core.results.parse_result.ParseResult`

解析结果的容器，实现 `IProblemDefinition` 接口。

```python
ParseResult(
    obj_analyzer,
    con_analyzer,
    var_manager,
    bound_manager,
    type_analyzer,
    functional_config=None
)
```

#### 属性

| 属性 | 说明 |
| --- | --- |
| `objective_funcs` | 解析后的目标 lambda 函数列表 |
| `objective_exprs` | 原始目标表达式列表 |
| `senses` | 优化方向列表，`'min'` / `'max'` |
| `objective_symbols` | 每个目标对应的符号变量集合列表 |
| `parsed_constraints` | 解析后的约束列表 |
| `discrete_constraints` | 离散变量相关约束 |
| `inequality_constraints` | 不等式约束 |
| `equality_constraints` | 等式约束 |
| `all_symbols` | 所有符号变量集合 |
| `sorted_symbols` | 排序后的符号列表 |
| `symbol_to_index` | 符号到索引的映射 |
| `n_variables` | 变量数量 |
| `variable_bounds` | 变量上下界元组 `(lower_bounds, upper_bounds)` |
| `discrete_variables` | 离散变量字典 |
| `pyomo_problem_type` | Pyomo 问题类型字符串 |
| `pymoo_problem_type` | Pymoo 问题类型字典 |
| `is_convex_qp` | 是否被识别为凸 QP |
| `ir_problem` | 统一 IR 问题对象 `IROptProblem` |
| `bound_manager` | 边界管理器实例 |

#### 方法

| 方法 | 说明 |
| --- | --- |
| `get_pyomo_problem_type()` | 返回 Pyomo 问题类型字符串 |
| `get_pymoo_problem_type()` | 返回 Pymoo 问题类型字典 |
| `has_integer_variables()` | 是否存在整数/离散变量 |

---

## 功能型配置

### `happymath.Opt.functional.config.ODEIVPConfig`

ODE/IVP（常微分方程初值问题）的功能型配置 dataclass。

```python
ODEIVPConfig(
    ode: list[sp.Eq],
    domain: DomainConfig,
    ivp_conds: dict = field(default_factory=dict),
    constants: dict = field(default_factory=dict),
    control: Optional[ControlParamConfig] = None,
    objective_meta: dict = field(default_factory=dict),
    constraint_meta: dict = field(default_factory=dict),
    extra_symbols: list = field(default_factory=list),
    bounds: dict = field(default_factory=dict),
    metrics: list = field(default_factory=list),
    param_symbols: list = field(default_factory=list),
    param_bounds: dict = field(default_factory=dict),
)
```

#### 字段说明

| 字段 | 说明 |
| --- | --- |
| `ode` | ODE 方程或方程组，SymPy `Eq` 列表 |
| `domain` | 连续域配置 `DomainConfig` |
| `ivp_conds` | 初值条件字典，如 `{x(0): 0}` |
| `constants` | 常数/系数字典 |
| `control` | 控制参数化配置 `ControlParamConfig` |
| `objective_meta` | 目标聚合元信息，如 `{0: {"aggregation": "integral", "expr": u(t)**2}}` |
| `constraint_meta` | 约束聚合元信息 |
| `extra_symbols` | 需要注册为决策变量的额外符号 |
| `bounds` | 针对 `extra_symbols` 的边界映射 `{symbol: (lb, ub)}` |
| `metrics` | 指标清单 `MetricSpec`；为空时由 `objective_meta`/`constraint_meta` 派生 |
| `param_symbols` | 额外标量参数符号 |
| `param_bounds` | 参数边界映射 |

---

### `happymath.Opt.functional.config.ODEBVPConfig`

ODE/BVP（常微分方程边值问题）的功能型配置 dataclass。

```python
ODEBVPConfig(
    ode: list[sp.Eq],
    domain: DomainConfig,
    bvp_conds: dict = field(default_factory=dict),
    constants: dict = field(default_factory=dict),
    control: Optional[ControlParamConfig] = None,
    metrics: list = field(default_factory=list),
    param_symbols: list = field(default_factory=list),
    param_bounds: dict = field(default_factory=dict),
)
```

#### 字段说明

| 字段 | 说明 |
| --- | --- |
| `ode` | ODE 方程或方程组 |
| `domain` | 连续域配置 |
| `bvp_conds` | 边界条件字典，如 `{x(t0): 0.0, x(t1): 1.0}` |
| `constants` | 常数/系数字典 |
| `control` | 控制参数化（当前 BVP 评估器不使用控制） |
| `metrics` / `param_symbols` / `param_bounds` | 与 `ODEIVPConfig` 一致 |

---

### `happymath.Opt.functional.config.DomainConfig`

连续域配置（当前仅支持 1D 时间域）。

```python
DomainConfig(var: sp.Symbol, t0: float, t1: float, grid_n: int = 101)
```

| 字段 | 说明 |
| --- | --- |
| `var` | 域变量符号，如 `t` |
| `t0` | 起始值 |
| `t1` | 终止值 |
| `grid_n` | 离散网格点数，默认 `101` |

---

### `happymath.Opt.functional.config.ControlParamConfig`

控制参数化配置。

```python
ControlParamConfig(
    kind: str = "piecewise_constant",
    func: Optional[sp.Function] = None,
    coeff_symbols: list = field(default_factory=list),
    segments: int = 10,
    bounds: Optional[tuple] = None,
)
```

| 字段 | 说明 |
| --- | --- |
| `kind` | 控制参数化类型，如 `"piecewise_constant"` |
| `func` | 控制函数符号，如 `u` |
| `coeff_symbols` | 系数符号列表，会注册为决策变量 |
| `segments` | 分段常数的段数 |
| `bounds` | 系数统一边界 `(lb, ub)` |

---

### `happymath.Opt.functional.config.MetricSpec`

功能型指标规格。

```python
MetricSpec(
    id: str,
    kind: Literal["integral", "terminal", "path"],
    integrand: Optional[IntegrandSpec] = None,
    state_index: Optional[int] = None,
    agg: str = "trapz",
)
```

| 字段 | 说明 |
| --- | --- |
| `id` | 唯一标识，如 `"obj:0"`、`"con:c1"` |
| `kind` | 指标类型：`"integral"` / `"terminal"` / `"path"` |
| `integrand` | `Integral` 类型时提供的被积函数规格 `IntegrandSpec` |
| `state_index` | 状态列索引，`terminal` / `path` 可用 |
| `agg` | 聚合方式：`"trapz"`、`"simpson"`、`"l2_norm"`、`"l1_norm"`、`"max_abs"`、`"mean_abs"` 等 |

---

### `happymath.Opt.functional.config.IntegrandSpec` / `WindowSpec`

辅助 dataclass：

- `IntegrandSpec(id, expr, window=None, channel=None)`：被积函数规格，`expr` 为 SymPy 表达式，`window` 为可选 `WindowSpec`。
- `WindowSpec(t0, t1)`：时间窗口 `[t0, t1]`，用于裁剪积分/路径指标范围。

---

## 返回值中的标准字段

无论 Pyomo 还是 Pymoo，每个结果字典通常包含以下字段：

| 字段 | 说明 |
| --- | --- |
| `algorithm` | 算法/求解器名称 |
| `result` | 原始求解器结果对象 |
| `success` | 是否成功 |
| `message` | 结果消息 |
| `exec_time` | 执行时间 |
| `solver_type` | `"pyomo"` 或 `"pymoo"` |
| `variables` | 解变量字典（Pyomo 结果通常直接提供） |
| `objective_value` | 目标值（Pyomo 结果通常直接提供） |
| `X` / `F` / `G` / `CV` | Pymoo 原始结果中的决策变量、目标值、约束值、约束违反度 |
| `n_evals` | Pymoo 评估次数 |
