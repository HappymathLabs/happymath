# DiffEq 微分方程

`happymath.DiffEq` 模块提供统一的微分方程建模与求解接口，支持常微分方程（ODE）的初值问题（IVP）和边值问题（BVP），以及偏微分方程（PDE）的数值求解。

## 模块结构

| 子模块/类 | 说明 |
|-----------|------|
| `happymath.DiffEq.ODE.ODEModule` | ODE 求解入口 |
| `happymath.DiffEq.PDE.PDEModule` | PDE 求解入口 |
| `happymath.DiffEq.PDE.adapters.solve_pde` | PDE 数值求解适配器 |

## ODE 求解流程

1. 使用 SymPy 定义 ODE 表达式
2. 创建 `ODEModule` 实例
3. 选择求解模式：
   - `.ode2scipy(mode, cond, const_cond)`：转换为 SciPy 可调用函数
   - `.ana_solve(ics, ...)`：符号解析解
   - `.num_solve(mode, cond, domain, ...)`：直接数值求解

## ODE 初值问题示例

```python
import sympy
import numpy as np
from scipy.integrate import solve_ivp
from happymath.DiffEq.ODE.ODEModule import ODEModule

# dy/dt = 2*y + t, y(0) = 1
t = sympy.symbols("t")
y = sympy.Function("y")
ode_expr = -y(t).diff(t, 1) + 2 * y(t) + t
ics = {y(0): 1}

ode_obj = ODEModule(ode_expr)
t_span = np.linspace(0, 5, 50)

# 转换为 SciPy 格式
func, y0, const = ode_obj.ode2scipy("IVP", ics)
sol = solve_ivp(func, (0, 5), y0, t_eval=t_span, args=const)
print("t=5 时 y ≈", sol.y[0, -1])
```

## ODE 解析解示例

```python
import sympy
from happymath.DiffEq.ODE.ODEModule import ODEModule

t = sympy.symbols("t")
y = sympy.Function("y")
ode_expr = -y(t).diff(t, 1) - y(t) + sympy.exp(-t)

ode_obj = ODEModule(ode_expr)
solution = ode_obj.ana_solve(ics={y(0): 1})
print(solution)
```

## ODE 数值解直接调用

```python
import sympy
import numpy as np
from happymath.DiffEq.ODE.ODEModule import ODEModule

y = sympy.Function("y")
t = sympy.symbols("t")
ode_expr = -y(t).diff(t, 1) - 3 * y(t) + sympy.cos(t)
ics = {y(0): 2}

de_obj = ODEModule(ode_expr)
t_domain = np.linspace(0, 3, 30)
result = de_obj.num_solve("IVP", ics, t_domain, solve_method="RK45", tol=0.001)
print("solution shape:", result.solution.shape)
print("success:", result.success)
```

## ODE 方程组示例

```python
import sympy
import numpy as np
from scipy.integrate import solve_ivp
from happymath.DiffEq.ODE.ODEModule import ODEModule

y1 = sympy.Function("y1")
y2 = sympy.Function("y2")
t = sympy.Symbol("t")

de1 = -y1(t).diff(t, 1) - y1(t) + 2 * y2(t)
de2 = -y2(t).diff(t, 1) + 3 * y1(t) - y2(t)
system = [de1, de2]

ics = {y1(0): 2, y2(0): 1}
sys_obj = ODEModule(system)
t_span = np.linspace(0, 3, 40)

func, y0, const = sys_obj.ode2scipy("IVP", ics)
sol = solve_ivp(func, (0, 3), y0, t_eval=t_span, args=const)
print("y1(t=3) ≈", sol.y[0, -1])
print("y2(t=3) ≈", sol.y[1, -1])
```

## PDE 数值求解示例

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import sympy as sp
from happymath.DiffEq.PDE.PDEModule import PDEModule
from happymath.DiffEq.PDE.adapters import solve_pde

x, t = sp.symbols("x t")
u = sp.Function("u")
D, v = sp.symbols("D v")

# 一维对流-扩散方程
expr = sp.Eq(
    sp.Derivative(u(x, t), t),
    D * sp.Derivative(u(x, t), (x, 2)) - v * sp.Derivative(u(x, t), x)
)

m = PDEModule(expr)
N = 64
state = np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))

res = solve_pde(
    m,
    state=state,
    t_range=0.002,
    dt=0.001,
    solver="explicit",
    const_cond={"D": 0.1, "v": 0.5},
    bc="periodic",
    grid_spec={"shape": (N,), "bounds": ((0, 1),), "periodic": True},
)
print("PDE 求解成功:", res.success)
```

## 关键 API

### `ODEModule`

- `__init__(sympy_obj, value_range="real")`：从 SymPy 表达式初始化
- `ode2scipy(mode, cond, const_cond=None)`：转换为 SciPy 函数
- `ana_solve(eq=None, ics=None, **kwargs)`：符号求解
- `num_solve(mode, cond, domain, ...)`：数值求解
- `stand_ode`：标准化后的 ODE 表达式

### `PDEModule`

- `__init__(sympy_obj, value_range="real", spatial_var_order=["x", "y"])`：初始化 PDE
- `stand_pde`：标准化后的 PDE 表达式
- `to_solvable_pde`：可求解格式
- `num_solve(state, t_range, dt, ...)`：数值求解

## 注意事项

- PDE 模块目前支持最多 2 个空间维度
- 不支持混合偏导数（如 ∂²u/∂x∂t）
- 边界条件通过 `bc` 参数传入，支持 `"periodic"` 或字典形式
- ODE 表达式建议写成等式形式或导数项在左侧的形式
