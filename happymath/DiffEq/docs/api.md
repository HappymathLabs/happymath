# DiffEq API 文档

本页按类/函数组织，说明 `happymath.DiffEq` 中 ODE、PDE 求解相关接口的参数与返回值。

## 1. 微分方程基础类

### `happymath.DiffEq.diffeq_core.de_base.DEBase`

ODE 与 PDE 模块的共同基类，负责表达式识别、符号假设与基础属性。

```python
DEBase(sympy_obj: Union[sympy.Expr, Iterable], value_range: str = "real")
```

#### 参数

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `sympy_obj` | `sympy.Expr` 或可迭代对象 | 微分方程表达式或方程组列表 |
| `value_range` | `str` | 符号假设，例如 `"real"`。若传入无效值会抛出 `ValueError` |

#### 属性

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `is_ode` | `bool` | 判断当前表达式是否为 ODE（委托给表达式分析器） |
| `is_pde` | `bool` | 判断当前表达式是否为 PDE（委托给表达式分析器） |
| `core_symbol` | `list` | 返回核心符号列表（自变量符号） |

---

## 2. ODE 求解入口

### `happymath.DiffEq.ODE.ODEModule.ODEModule`

ODE 求解主类，基于组件化架构，无全局状态。

```python
ODEModule(sympy_obj: Union[sympy.Expr, list], value_range: str = "real")
```

#### 参数

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `sympy_obj` | `sympy.Expr` 或 `list` | SymPy ODE 表达式或表达式列表 |
| `value_range` | `str` | 变量符号假设，默认 `"real"` |

#### 属性

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `expr` | `sympy.Expr` / `list` | 当前表达式；修改后会自动使内部缓存失效 |
| `stand_ode` | `list` | 标准化后的 ODE 表达式（带缓存） |
| `show_stand_ode` | `None` | 在 Jupyter/IPython 中以富文本形式展示标准化结果 |
| `Y_symbols` | `list[sympy.Symbol]` | 标准化过程中使用的替换符号 |

#### 方法

##### `ode2scipy(mode, cond, const_cond=None)`

将 SymPy ODE 转换为 SciPy 可调用函数。

```python
ode2scipy(
    mode: str,
    cond: Dict,
    const_cond: Optional[Dict] = None
)
```

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `mode` | `str` | 求解模式，`'IVP'` 或 `'BVP'` |
| `cond` | `Dict` | 初始/边界条件字典 |
| `const_cond` | `Optional[Dict]` | 可选常数字典 |

**返回值**：SciPy 兼容函数、替换字典、常数值元组。

##### `ana_solve(eq=None, ics=None, **kwargs)`

调用 SymPy `dsolve` 求解析解。

```python
ana_solve(
    eq: Optional[Union[sympy.Expr, list]] = None,
    ics: Optional[Dict] = None,
    **kwargs
) -> Union[sympy.Eq, List[sympy.Eq]]
```

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `eq` | `sympy.Expr` / `list` / `None` | 待求解方程，默认使用当前 `self.expr` |
| `ics` | `Optional[Dict]` | 初始条件字典，传给 `dsolve` |
| `**kwargs` | — | 额外关键字参数，转发给 `dsolve` |

**返回值**：SymPy `Eq` 或 `Eq` 列表。

**注意**：并非所有方程都能求出解析解，失败时会抛出 `SolverExecutionError`。

##### `num_solve(mode, cond, domain, const_cond=None, bc=None, init_guess="linear", solve_method="RK45", tol=0.001, bc_tol=None)`

ODE 数值求解入口。

```python
num_solve(
    mode: str,
    cond: Dict,
    domain: np.ndarray,
    const_cond: Optional[Dict] = None,
    bc: Optional[Callable] = None,
    init_guess: Union[str, np.ndarray] = "linear",
    solve_method: str = "RK45",
    tol: float = 0.001,
    bc_tol: Optional[float] = None
) -> np.ndarray
```

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `mode` | `str` | `'IVP'` 或 `'BVP'` |
| `cond` | `Dict` | 初始/边界条件 |
| `domain` | `np.ndarray` | 自变量采样点，例如 `np.linspace(0, 5, 50)` |
| `const_cond` | `Optional[Dict]` | 常数字典 |
| `bc` | `Optional[Callable]` | 边界条件函数，BVP 必需 |
| `init_guess` | `str` / `np.ndarray` | BVP 初始猜测策略，`"linear"` 表示全 1 数组，也可传入数组 |
| `solve_method` | `str` | SciPy 方法，如 `"RK45"`、`"RK23"`、`"DOP853"`、`"Radau"`、`"BDF"`、`"LSODA"` |
| `tol` | `float` | 相对容差 |
| `bc_tol` | `Optional[float]` | 边界条件容差，默认与 `tol` 相同 |

**返回值**：形状为 `(n_points, n_states)` 的解数组。

**内部流程**：
- IVP 调用 `_solve_ivp`，使用 `scipy.integrate.solve_ivp`。
- BVP 调用 `_solve_bvp`，使用 `scipy.integrate.solve_bvp`。

---

## 3. ODE 结果容器

### `happymath.DiffEq.ODE.core.result.ODESolutionResult`

```python
@dataclass
ODESolutionResult(
    domain: np.ndarray,
    solution: np.ndarray,
    error: Union[np.ndarray, List[float]],
    solution_func: Callable,
    substitution_dict: Dict[Any, Any],
    success: bool,
    message: str = ""
)
```

#### 属性

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `domain` | `np.ndarray` | 自变量采样点 |
| `solution` | `np.ndarray` | 解值，形状 `(n_points, n_states)` |
| `error` | `np.ndarray` / `List[float]` | 局部误差估计或容差占位列表 |
| `solution_func` | `Callable` | 连续解可调用对象 `f(t)`，若可用 |
| `substitution_dict` | `Dict[Any, Any]` | 求解输入映射，如初始条件、常数 |
| `success` | `bool` | 求解器是否报告成功 |
| `message` | `str` | 状态信息 |

---

## 4. PDE 求解入口

### `happymath.DiffEq.PDE.PDEModule.PDEModule`

PDE 求解主类。

```python
PDEModule(
    sympy_obj,
    value_range: str = "real",
    spatial_var_order=["x", "y"]
)
```

#### 参数

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `sympy_obj` | `sympy.Expr` / `list` | SymPy PDE 表达式或表达式列表 |
| `value_range` | `str` | 符号假设，默认 `"real"` |
| `spatial_var_order` | `list[str]` | 空间变量优先级，默认 `["x", "y"]` |

#### 当前限制

- 空间维度大于 2 时不支持。
- 不支持混合偏导数（如 `∂²u/∂x∂t`）。

#### 属性

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `expr` | `sympy.Expr` / `list` | 当前表达式；修改后自动失效缓存 |
| `stand_pde` | `list` | 标准化后的 PDE 表达式（带缓存） |
| `to_solvable_pde` | `Dict[str, str]` | 可提交给 `py-pde` 的右端项字典 |
| `spatial_var_list` | `list[str]` | 检测到的空间变量名 |
| `time_var` | `sympy.Symbol` | 检测到的时间变量 `t` |

#### 方法

##### `ana_solve()`

```python
ana_solve()
```

PDE 解析解入口。当前实现为 `pass`，即**未实现**。

##### `num_solve(state, t_range, dt, const_cond=None, solver="explicit", bc=None, bc_ops=None, grid_spec=None)`

PDE 数值求解 facade 接口。

```python
num_solve(
    state,
    t_range,
    dt,
    const_cond: dict = None,
    solver: str = "explicit",
    bc: dict | str | None = None,
    bc_ops: dict | None = None,
    grid_spec: dict | None = None
)
```

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `state` | `py-pde Field` / `np.ndarray` / `Dict[str, np.ndarray]` | 初始场；numpy 数组会通过 `grid_spec` 自动构建 `ScalarField` |
| `t_range` | `Any` | 时间范围，如 `(0, 1)` 或 `1.0` |
| `dt` | `float` | 时间步长 |
| `const_cond` | `dict` / `None` | 常数/系数字典；若表达式含自由常数则必须提供 |
| `solver` | `str` | 转发给 `py-pde` 的求解器名，默认 `"explicit"` |
| `bc` | `dict` / `str` / `None` | 边界条件，如 `"periodic"` 或 `{"x-": {"value": 0}}` |
| `bc_ops` | `dict` / `None` | 算子级边界条件，转发给 `py-pde` |
| `grid_spec` | `dict` / `None` | numpy 状态对应的网格说明，如 `{"bounds": ((0, 1),), "shape": (64,), "periodic": True}` |

**返回值**：`py-pde` 的 `Trajectory` 解对象。

---

## 5. PDE 结果容器

### `happymath.DiffEq.PDE.core.result.PDESolutionResult`

```python
@dataclass
PDESolutionResult(
    solution: Any,
    time_range: Any,
    dt: float,
    solver: str,
    constants: Dict[str, Any],
    rhs: Dict[str, Any],
    success: bool,
    message: str = ""
)
```

#### 属性

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `solution` | `Any` | 后端返回的解对象，例如 `py-pde Trajectory` |
| `time_range` | `Any` | 积分使用的时间范围 |
| `dt` | `float` | 时间步长 |
| `solver` | `str` | 使用的求解器标识 |
| `constants` | `Dict[str, Any]` | 传入 PDE 的常数映射 |
| `rhs` | `Dict[str, Any]` | 右端项表示，可能是字符串表达式或函数 |
| `success` | `bool` | 求解器是否报告成功 |
| `message` | `str` | 状态信息 |

---

## 6. PDE 求解适配器

### `happymath.DiffEq.PDE.adapters.pde_adapter.solve_pde`

`PDEModule.num_solve` 内部调用的适配器函数，负责将标准化结果构建为 `py-pde.PDE` 并求解。

```python
solve_pde(
    ctx,
    state: Any,
    t_range: Any,
    dt: float,
    solver: str = "explicit",
    const_cond: Optional[Dict] = None,
    bc: Optional[dict | str] = None,
    bc_ops: Optional[Dict] = None,
    grid_spec: Optional[Dict] = None
) -> PDESolutionResult
```

#### 参数

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `ctx` | `PDEModule` | PDE 模块实例 |
| `state` | `Any` | 初始场 |
| `t_range` | `Any` | 时间范围 |
| `dt` | `float` | 时间步长 |
| `solver` | `str` | `py-pde` 求解器名 |
| `const_cond` | `Optional[Dict]` | 常数字典；若 `ctx.free_consts` 非空则必须提供 |
| `bc` | `Optional[dict \| str]` | 边界条件 |
| `bc_ops` | `Optional[Dict]` | 算子级边界条件 |
| `grid_spec` | `Optional[Dict]` | 网格说明 |

**返回值**：`PDESolutionResult`。

#### 内部处理步骤

1. 从 `ctx.to_solvable_pde` 获取可求解格式。
2. 若表达式含自由常数而 `const_cond` 为空，抛出 `MissingParameterError`。
3. 通过 `_prepare_state` 将 numpy/字典转换为 `py-pde Field`。
4. 通过 `_prepare_consts` 将标量/数组/向量转换为 `py-pde` 可接受的常量。
5. 尝试使用字符串右端项；若无法表达则回退到 `_make_function_rhs`。
6. 构建 `pde.PDE` 并调用 `solve`。

#### 内部辅助函数

| 函数 | 说明 |
| --- | --- |
| `_prepare_state(state, grid_spec)` | 将 `Field` / `np.ndarray` / `Dict[str, np.ndarray]` 转换为 `py-pde FieldBase` |
| `_prepare_consts(consts, state_field)` | 将标量、`ndarray`、向量转换为 `py-pde` 标量场/向量场 |
| `_maybe_rewrite_rhs_strings(rhs_map, state_field)` | 在 1D 情况下将 `d2_dx2` 改写为 `laplace` 等 |
| `_make_function_rhs(ctx, state_field, consts, bc)` | 对含一阶导或复杂项的 PDE 构建函数右端项 |
| `_inject_basis_vectors(consts, rhs_map, state_field)` | 当字符串右端项引用 `ex`/`ey`/`ez` 时自动注入基向量 |

---

## 7. 异常类型（简要）

`happymath.DiffEq.diffeq_core.de_exceptions` 中定义了以下常用异常，文档中可能遇到：

| 异常 | 触发场景 |
| --- | --- |
| `DEException` | 所有微分方程异常的基类 |
| `InvalidExpressionError` | 表达式不是有效 ODE/PDE |
| `SolverNotFoundError` | 请求了不存在的求解器 |
| `SolverExecutionError` | 求解器执行失败 |
| `InvalidParameterError` | 参数非法，如 `mode` 不是 `'IVP'`/`'BVP'` |
| `MissingParameterError` | 缺少必需参数，如 BVP 缺少 `bc` |
| `BoundaryConditionError` | 边界条件格式错误 |
| `ExpressionStandardizationError` | 表达式标准化失败 |
