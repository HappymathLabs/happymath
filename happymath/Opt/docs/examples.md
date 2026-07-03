# Opt 模块案例演示

以下所有代码均已在 `happymath` 环境中实际运行通过。对于进化算法示例，已主动降低评估预算以节省运行时间。

---

## 1. 单目标 Pymoo 箱约束优化

使用 Pymoo 的遗传算法（GA）求解一个简单的二次函数最小化问题。

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

print("success:", res.success)
print("variables:", res.variables)
print("objective_value:", res.objective_value)
```

输出示例：

```text
success: True
variables: {'x': 1.9901296032203561, 'y': 2.9477296805200592}
objective_value: 0.0028296110311226788
```

---

## 2. 单目标 Pyomo 线性规划

以下线性规划使用 `appsi_highs` 求解器（需先安装 `highspy`）。
当前版本中，为避免 `BoundManager` 自动紧化触发内部错误，建议显式传入 `tighten_bounds="none"`。

```python
import sympy as sp
from happymath.Opt.OptModule import OptModule

x, y = sp.symbols('x y', real=True)

obj = {"max": 3*x + 4*y}
constraints = [
    x >= 0,
    y >= 0,
    x + 2*y <= 14,
    3*x - y >= 0,
    x <= 10,
]

opt = OptModule(
    obj_func=obj,
    constraints=constraints,
    mode="pyomo",
    tighten_bounds="none",
)
res = opt.solve(solver="appsi_highs", use_auto_solvers=False, max_solvers=1)

print("success:", res.success)
print("variables:", res.variables)
print("objective_value:", res.objective_value)
```

输出示例：

```text
success: True
variables: {'x': 10.0, 'y': 2.0}
objective_value: 38.0
```

> 提示：若本机未安装 `highspy`，可执行 `pip install highspy`，随后使用 `solver="appsi_highs"`。其他常见 Pyomo 求解器如 `cbc`、`glpk`、`ipopt` 需要单独安装可执行文件。

---

## 3. 多目标 NSGA2 优化并查看 Pareto 前沿

多个最小化目标以列表形式传入 `{"min": [expr1, expr2]}`。多目标模式强制使用 Pymoo，且所有变量必须有界。

```python
import sympy as sp
from happymath.Opt.OptModule import OptModule

x, y = sp.symbols('x y', real=True)

obj = {
    "min": [
        (x - 1)**2 + y**2,
        (x + 1)**2 + y**2,
    ]
}
constraints = [
    x >= -3, x <= 3,
    y >= -3, y <= 3,
]

opt = OptModule(obj_func=obj, constraints=constraints, mode="pymoo")
opt.pymoo_solver._budget_override = 300  # 降低评估预算

res = opt.solve(solver="NSGA2", use_auto_solvers=False, max_solvers=1)

print("success:", res.success)
print("best variables:", res.variables)
print("best objective_value:", res.objective_value)
print("pareto_front size:", len(res.pareto_front))
print("first front point:", res.pareto_front[0])
```

输出示例：

```text
success: True
best variables: {'x': -0.08538928940326973, 'y': 0.37271424867123304}
best objective_value: [1.3169858207138965, 0.9754286631008177]
pareto_front size: 17
first front point: {'objectives': [1.7021814917689808, 0.5240360021964305], 'variables_array': [-0.29453637239313757, 0.16234861354565577], 'algorithm': 'NSGA2'}
```

---

## 4. 最优控制问题：ODE + 控制参数化

系统动态：

```
x'(t) = -x(t) + u(t),   x(0) = 1
```

控制 `u(t)` 采用分段常数参数化，5 个系数 `c0~c4` 为决策变量；目标是最小化控制能量的积分，并要求终端状态 `x(1) = 0`。

```python
import sympy as sp
from happymath.Opt.OptModule import OptModule
from happymath.Opt.functional.config import ODEIVPConfig, DomainConfig, ControlParamConfig

t = sp.symbols('t', real=True)
x = sp.Function("x")
u = sp.Function("u")
coeffs = sp.symbols("c0:5", real=True)

# 功能型配置
func_cfg = ODEIVPConfig(
    ode=[sp.Eq(sp.diff(x(t), t, 1), -x(t) + u(t))],
    domain=DomainConfig(var=t, t0=0.0, t1=1.0, grid_n=101),
    ivp_conds={x(0): 1.0},
    control=ControlParamConfig(
        kind="piecewise_constant",
        func=u,
        coeff_symbols=list(coeffs),
        segments=5,
        bounds=(-2.0, 2.0),
    ),
    objective_meta={0: {"aggregation": "integral", "expr": u(t)**2}},
    constraint_meta={
        "c_term": {
            "aggregation": "final_state",
            "expr": x(t),
            "sense": "eq",
            "state_index": 0,
        }
    },
    extra_symbols=list(coeffs),
    bounds={s: (-2.0, 2.0) for s in coeffs},
)

# 目标占位表达式，实际指标由 func_cfg 中的 objective_meta 计算
obj = {"min": sp.integrate(u(t)**2, (t, 0, 1))}

# 为控制系数补充显式边界，确保 Pymoo 严格有界检查通过
constraints = [c <= 2.0 for c in coeffs] + [c >= -2.0 for c in coeffs]

opt = OptModule(
    obj_func=obj,
    constraints=constraints,
    mode="pymoo",
    default_search_range=2.0,
    functional_config=func_cfg,
)

# 降低进化算法评估预算
opt.pymoo_solver._budget_override = 400

res = opt.solve(solver="GA", use_auto_solvers=False, max_solvers=1)

print("success:", res.success)
print("variables:", res.variables)
print("objective_value:", res.objective_value)
```

输出示例：

```text
success: True
variables: {'c0': -1.5992221491977183, 'c1': -1.430885337248241, 'c2': -0.5867315950390224, 'c3': -0.35075183739415305, 'c4': 0.15526693601342778}
objective_value: 1.0065995753562702
```

> 说明：该案例属于 FUNCTIONAL（功能型）优化，求解器在每次评估时都会数值积分 ODE，因此比纯代数优化慢。实际使用中可根据精度需求调整 `grid_n` 与 `_budget_override`。
