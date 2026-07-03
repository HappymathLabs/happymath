# Opt 数学优化

`happymath.Opt` 模块是一个统一的数学优化框架，支持代数优化、最优控制以及基于微分方程的优化问题。底层集成了 Pyomo（精确/数学规划求解器）和 Pymoo（进化/启发式算法）两大求解后端。

## 模块结构

| 类/模块 | 说明 |
|---------|------|
| `happymath.Opt.OptModule.OptModule` | 优化问题主入口 |
| `happymath.Opt.OptBase` | 优化基类，负责表达式解析 |
| `happymath.Opt.solvers.pyomo_solver.PyomoSolver` | Pyomo 求解器封装 |
| `happymath.Opt.solvers.pymoo_solver.PymooSolver` | Pymoo 进化算法封装 |
| `happymath.Opt.results.opt_result.OptResult` | 统一结果容器 |

## 基本用法

优化问题通过 `OptModule` 创建，目标函数使用字典形式：

```python
{"min": expr}  # 最小化 expr
{"max": expr}  # 最大化 expr
```

约束条件使用 SymPy 关系式列表，例如 `[x >= 0, x <= 10]`。

## 单目标代数优化示例

```python
import sympy as sp
from happymath.Opt.OptModule import OptModule

x1, x2 = sp.symbols("x1 x2", real=True)
obj = {"min": (x1 - 1) ** 2 + (x2 - 2) ** 2}
constraints = [x1 >= -5, x1 <= 5, x2 >= -5, x2 <= 5]

opt = OptModule(obj, constraints, mode="pymoo", default_search_range=5.0)
res = opt.solve(solver="GA", use_auto_solvers=False, max_solvers=1)

print("求解成功:", res.success)
print("最优变量:", res.variables)
print("最优目标值:", res.objective_value)
```

## Pyomo 精确求解示例

```python
import sympy as sp
from happymath.Opt.OptModule import OptModule

x, y = sp.symbols("x y", real=True)
obj = {"min": 3 * x + 4 * y}
constraints = [x + 2 * y >= 8, 3 * x + y >= 9, x >= 0, y >= 0]

opt = OptModule(obj, constraints, mode="pyomo")
res = opt.solve()
print("最优解:", res.variables)
print("最优值:", res.objective_value)
```

## 多目标优化示例

```python
import sympy as sp
from happymath.Opt.OptModule import OptModule

x1, x2 = sp.symbols("x1 x2", real=True)
# 同时最小化两个目标
obj = {"min": [x1 ** 2 + x2 ** 2, (x1 - 1) ** 2 + (x2 - 1) ** 2]}
constraints = [x1 >= -2, x1 <= 2, x2 >= -2, x2 <= 2]

opt = OptModule(obj, constraints, mode="pymoo", default_search_range=2.0)
res = opt.solve(solver="NSGA2", use_auto_solvers=False, max_solvers=1)
print("帕累托前沿:", res.pareto_front[:3])
```

## 最优控制问题示例

```python
import sympy as sp
from happymath.Opt.OptModule import OptModule
from happymath.Opt.functional.config import (
    ODEIVPConfig, DomainConfig, ControlParamConfig
)

t = sp.symbols("t", real=True)
x = sp.Function("x")
u = sp.Function("u")

ode = [sp.Eq(sp.diff(x(t), t, 1), -x(t) + u(t))]
coeffs = sp.symbols("c0:5", real=True)

func_cfg = ODEIVPConfig(
    ode=ode,
    domain=DomainConfig(var=t, t0=0.0, t1=1.0, grid_n=101),
    ivp_conds={x(0): 1.0},
    control=ControlParamConfig(
        kind="piecewise_constant",
        func=u,
        coeff_symbols=list(coeffs),
        segments=5,
        bounds=(-2.0, 2.0),
    ),
    objective_meta={0: {"aggregation": "integral", "expr": u(t) ** 2}},
    constraint_meta={
        "c_term": {"aggregation": "final_state", "expr": x(t), "sense": "eq", "state_index": 0}
    },
    extra_symbols=list(coeffs),
    bounds={},
)

obj = {"min": sp.integrate(u(t) ** 2, (t, 0, 1))}
constraints = []
for c in coeffs:
    constraints.extend([c <= 2.0, c >= -2.0])

opt = OptModule(
    obj_func=obj,
    constraints=constraints,
    mode="pymoo",
    default_search_range=2.0,
    functional_config=func_cfg,
)
res = opt.solve(solver="GA", use_auto_solvers=False, max_solvers=1)
print("控制系数:", res.variables)
print("目标值:", res.objective_value)
```

## 关键 API

### `OptModule`

- `__init__(obj_func, constraints=None, mode="auto", default_search_range=100, ...)`：初始化优化问题
- `solve(solver=None, use_auto_solvers=True, max_solvers=3, ref=None)`：执行求解

### `OptResult`

- `success`：是否求解成功
- `variables`：最优解变量字典
- `objective_value`：最优目标值（单目标为 float，多目标为 list）
- `solution`：包含变量和目标值的字典
- `all_solutions`：所有成功解的列表
- `pareto_front`：多目标优化的帕累托前沿
- `raw_all_solutions`：原始求解器返回结果

## 求解模式说明

| `mode` | 说明 |
|--------|------|
| `"auto"` | 根据问题类型自动选择 Pyomo/Pymoo |
| `"pyomo"` | 仅使用 Pyomo，适合线性/非线性规划 |
| `"pymoo"` | 仅使用 Pymoo，适合启发式/多目标优化 |

## 注意事项

- Pymoo 模式要求所有变量都有显式上下界
- 多目标优化目前仅支持 Pymoo 后端
- 微分代数优化问题需要通过 `functional_config` 传入配置
- 可以通过 `solver` 参数指定具体算法，如 `"GA"`、`"NSGA2"`、`"cbc"`、`"glpk"` 等
