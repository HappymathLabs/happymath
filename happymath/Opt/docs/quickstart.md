# Opt 模块快速开始

## 功能定位

`happymath.Opt` 是一个统一的数学优化框架。它的核心思想是：

- 用 SymPy 符号表达式直接描述目标函数与约束；
- 自动识别问题类型（LP、QP、NLP、MILP、多目标、含 ODE 的最优控制等）；
- 在底层统一对接 **Pyomo**（精确/梯度求解器）与 **Pymoo**（进化/启发式算法）；
- 返回统一的 `OptResult` 结果对象，无需关心底层求解器差异。

## 主要类

| 类 | 路径 | 作用 |
| --- | --- | --- |
| `OptModule` | `happymath.Opt.OptModule.OptModule` | 用户主入口，负责建模、选择求解库并调用求解 |
| `OptBase` | `happymath.Opt.opt_core.opt_base.OptBase` | 基类，调用表达式处理器并缓存解析结果 |
| `OptResult` | `happymath.Opt.results.opt_result.OptResult` | 统一结果包装器，支持单/多目标、Pareto 前沿 |
| `PyomoSolver` | `happymath.Opt.solvers.pyomo_solver.PyomoSolver` | Pyomo 求解器后端，处理 LP/QP/NLP/MILP/DAE 等 |
| `PymooSolver` | `happymath.Opt.solvers.pymoo_solver.PymooSolver` | Pymoo 进化算法后端，处理单/多目标黑盒问题 |
| `ExpressionProcessor` | `happymath.Opt.opt_expr.processor.ExpressionProcessor` | 表达式处理器入口，生成 `ParseResult` |
| `ParseResult` | `happymath.Opt.opt_expr.core.results.parse_result.ParseResult` | 解析结果容器，包含变量、边界、IR 问题、问题类型等 |

## 主要方法

### `OptModule.__init__`

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

### `OptModule.solve`

```python
result = opt.solve(
    solver=None,
    use_auto_solvers=True,
    max_solvers=3,
    ref=None,
)
```

### `OptResult` 常用属性

| 属性 | 说明 |
| --- | --- |
| `success` | 是否存在至少一个成功结果 |
| `variables` | 最优解的变量字典 `{name: value}` |
| `objective_value` | 单目标为 `float`，多目标为 `List[float]` |
| `solution` | 包含 `variables` 与 `objective_value` 的字典 |
| `pareto_front` | 多目标优化返回的 Pareto 前沿列表 |
| `all_solutions` | 所有求解器返回的解列表 |
| `all_solvers` | 所有求解器运行信息 |
| `message` | 求解消息 |

## 关键参数说明

- `obj_func`：目标函数字典，格式 `{"min": expr}` 或 `{"max": expr}`，多目标可传 `{"min": [expr1, expr2]}`。
- `constraints`：约束列表，如 `[x >= 0, x + y <= 1]`。
- `mode`：求解库选择，`"auto"` / `"pyomo"` / `"pymoo"`。
  - `"auto"`：单目标且所有变量有界时同时启用 Pyomo 与 Pymoo，否则回退到 Pyomo。
  - `"pyomo"`：仅使用 Pyomo。
  - `"pymoo"`：仅使用 Pymoo，要求所有变量必须有界。
- `default_search_range`：当变量未显式给出上下界时，默认搜索区间为 `[-range, range]`。
- `solver`：`solve()` 中指定的求解器/算法名，如 `"cbc"`、`"glpk"`、`"ipopt"`、`"GA"`、`"NSGA2"`。
- `use_auto_solvers`：是否自动尝试多个求解器。
- `max_solvers`：最大尝试求解器数量。

## 当前已知限制

1. **Pymoo 模式要求所有变量有界**：若存在无界变量会报错或自动禁用 Pymoo。
2. **多目标仅支持 Pymoo**：`mode="pyomo"` 对多目标会直接报错。
3. **functional 配置使用门槛高**：ODE/IVP、BVP、PDE 最优控制需要手动构造 `ODEIVPConfig` / `ODEBVPConfig` / `PDEConfig`。
4. **部分 Pyomo 求解器需额外安装**：如 `cbc`、`glpk`、`ipopt`、`scip` 等为外部可执行程序；本机若无可用求解器，Pyomo 后端会失败。若已安装 `highspy`，可尝试 `solver="appsi_highs"`。
5. **进化算法结果不稳定**：Pymoo 算法依赖随机种子与评估预算，结果可能随运行略有不同。
6. **Pyomo.DAE 功能型问题对变量指数形式支持有限**：例如 `x**alpha`（alpha 为变量）会提示改用 Pymoo。
7. **边界自动紧化存在缺陷**：当前版本中，对包含线性约束的问题使用默认 `tighten_bounds` 可能触发 `BoundManager` 内部错误；临时解决方法是显式传入 `tighten_bounds="none"`。

## 最简单可运行案例：单目标箱约束优化

```python
import sympy as sp
from happymath.Opt.OptModule import OptModule

x, y = sp.symbols('x y', real=True)

obj = {"min": (x - 2)**2 + (y - 3)**2}
constraints = [
    x >= -10, x <= 10,
    y >= -10, y <= 10,
]

opt = OptModule(obj_func=obj, constraints=constraints, mode="pymoo")
res = opt.solve(solver="GA", use_auto_solvers=False, max_solvers=1)

print(res.success)          # True
print(res.variables)        # {'x': ~2.0, 'y': ~3.0}
print(res.objective_value)  # 接近 0
```

运行结果（近似）：

```text
True
{'x': 1.9901296032203561, 'y': 2.9477296805200592}
0.0028296110311226788
```
