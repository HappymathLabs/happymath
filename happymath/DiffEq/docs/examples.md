# DiffEq 案例演示

本章给出 `happymath.DiffEq` 中 ODE 与 PDE 主要接口的完整可运行示例。所有代码均已在当前 happymath 环境下实际运行通过。

## 1. ODE IVP 一阶方程

求解初值问题 `dy/dt = 2*y + t, y(0) = 1`。

```python
import numpy as np
import sympy as sp
from happymath.DiffEq.ODE.ODEModule import ODEModule

y = sp.Function("y")
t = sp.symbols("t")

ode_expr = -y(t).diff(t, 1) + 2 * y(t) + t
ics = {y(0): 1}

ode_obj = ODEModule(ode_expr)
domain = np.linspace(0, 5, 50)
solution = ode_obj.num_solve(mode="IVP", cond=ics, domain=domain)

print("solution shape:", solution.shape)
print(solution[:5])
```

输出示例：

```text
solution shape: (50, 1)
[[1.        ]
 [1.23199343]
 [1.52810256]
 [1.89642349]
 [2.34493496]]
```

## 2. ODE IVP 方程组

求解方程组：

```
dy1/dx = 2*y1 + y2
 dy2/dx = y1 + 3*y2
y1(0) = 1, y2(0) = 0
```

```python
import numpy as np
import sympy as sp
from happymath.DiffEq.ODE.ODEModule import ODEModule

y1 = sp.Function('y1')
y2 = sp.Function('y2')
x = sp.Symbol('x')

eq1 = -y1(x).diff(x, 1) + 2 * y1(x) + y2(x)
eq2 = -y2(x).diff(x, 1) + y1(x) + 3 * y2(x)

sys_obj = ODEModule([eq1, eq2])
ics = {y1(0): 1, y2(0): 0}
domain = np.linspace(0, 2, 50)

sol = sys_obj.num_solve(mode="IVP", cond=ics, domain=domain, solve_method="RK45")
print("system solution shape:", sol.shape)
print(sol[:3])
```

输出示例：

```text
system solution shape: (50, 2)
[[1.         0.        ]
 [1.08728513 0.04094333]
 [1.19034631 0.08816283]]
```

## 3. ODE 解析解

使用 `ana_solve` 调用 SymPy `dsolve` 求解析解。

```python
import sympy as sp
from happymath.DiffEq.ODE.ODEModule import ODEModule

y = sp.Function("y")
t = sp.symbols("t")

ode_expr = -y(t).diff(t, 1) - y(t)
ana_obj = ODEModule(ode_expr)
ana_sol = ana_obj.ana_solve(ics={y(0): 1})
print(ana_sol)
```

输出：

```text
Eq(y(t), exp(-t))
```

> 注意：复杂或非线性方程可能无法得到解析解，此时会抛出异常。

## 4. ODE num_solve 直接调用

`num_solve` 内部会调用 `ode2scipy` 与 `scipy.integrate.solve_ivp`，最终返回解数组。可以指定 `solve_method`、`tol` 等参数。

```python
import numpy as np
import sympy as sp
from happymath.DiffEq.ODE.ODEModule import ODEModule

y = sp.Function("y")
t = sp.symbols("t")

ode_expr = -y(t).diff(t, 1) - 3 * y(t) + sp.cos(t)
ics = {y(0): 2}

ode_obj = ODEModule(ode_expr)
domain = np.linspace(0, 3, 30)

sol = ode_obj.num_solve(
    mode="IVP",
    cond=ics,
    domain=domain,
    solve_method="RK45",
    tol=1e-6,
)
print("direct num_solve shape:", sol.shape)
print(sol[:3])
```

## 5. PDE 一维对流扩散方程数值求解

求解：

```
∂u/∂t = D * ∂²u/∂x² - v * ∂u/∂x
```

初始场为 `sin(x)`，周期边界。

```python
import numpy as np
import sympy as sp
from happymath.DiffEq.PDE.PDEModule import PDEModule

x, t = sp.symbols('x t')
u = sp.Function('u')
D, v = sp.symbols('D v')

pde_expr = sp.Eq(
    sp.Derivative(u(x, t), t),
    D * sp.Derivative(u(x, t), (x, 2)) - v * sp.Derivative(u(x, t), x)
)

pde_obj = PDEModule(pde_expr)

N = 64
state = np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))

traj = pde_obj.num_solve(
    state=state,
    t_range=0.002,
    dt=0.001,
    solver='explicit',
    const_cond={'D': 0.1, 'v': 0.5},
    bc='periodic',
    grid_spec={'shape': (N,), 'bounds': ((0, 1),), 'periodic': True},
)

print("PDE trajectory:", traj)
print("final field shape:", traj.data.shape)
```

输出示例：

```text
PDE trajectory: ScalarField(grid=CartesianGrid(...), data=Array(64,))
final field shape: (64,)
```

> 注意：PDE 求解依赖 `py-pde`。如果本地 `py-pde` 版本过新，可能出现 `module 'pde' has no attribute 'FieldBase'` 等兼容性问题，需要安装与当前代码兼容的版本（例如 `py-pde<0.50`）。

## 6. 查看标准化方程

无论是 ODE 还是 PDE，都可以查看自动标准化后的表达式，以确认内部转换是否正确。

```python
# ODE
from happymath.DiffEq.ODE.ODEModule import ODEModule
import sympy as sp

y = sp.Function("y")
t = sp.symbols("t")
ode = ODEModule(-y(t).diff(t, 1) + 2 * y(t) + t)
print(ode.stand_ode)

# PDE
from happymath.DiffEq.PDE.PDEModule import PDEModule

x, t = sp.symbols('x t')
u = sp.Function('u')
pde = PDEModule(sp.Eq(sp.Derivative(u(x, t), t), sp.Derivative(u(x, t), (x, 2))))
print(pde.stand_pde)
print(pde.to_solvable_pde)
```

> 在 Jupyter 环境中，ODE 还可以使用 `ode_obj.show_stand_ode` 以富文本方式展示标准化结果。
