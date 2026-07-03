# DiffEq 快速开始

`happymath.DiffEq` 是 HappyMath 库中用于求解微分方程的统一入口模块，覆盖常微分方程（ODE）与偏微分方程（PDE）两类问题。核心设计目标是用 SymPy 表达式描述方程，再自动转换到 SciPy（ODE）或 py-pde（PDE）进行数值求解，同时保留符号解析能力。

## 1. 模块定位

- **ODE**：支持一阶/高阶方程、方程组、初值问题（IVP）与边值问题（BVP）的数值求解，以及部分方程的解析解。
- **PDE**：支持基于 `py-pde` 的数值求解，输入方程后自动转换为可求解格式。

## 2. 主要类

| 类 | 说明 |
| --- | --- |
| `happymath.DiffEq.ODE.ODEModule.ODEModule` | ODE 求解入口类 |
| `happymath.DiffEq.ODE.core.result.ODESolutionResult` | ODE 数值解结果容器 |
| `happymath.DiffEq.PDE.PDEModule.PDEModule` | PDE 求解入口类 |
| `happymath.DiffEq.PDE.core.result.PDESolutionResult` | PDE 数值解结果容器 |

## 3. 主要方法

### ODE

| 方法 | 说明 |
| --- | --- |
| `ode2scipy(mode, cond, const_cond)` | 将 SymPy ODE 转换为 SciPy 可调用函数与初始/边界参数 |
| `ana_solve(eq, ics, **kwargs)` | 调用 SymPy `dsolve` 求解析解 |
| `num_solve(mode, cond, domain, ...)` | 直接调用 SciPy 求解，返回解数组 |
| `stand_ode`（属性） | 标准化后的 ODE 表达式 |

### PDE

| 方法/属性 | 说明 |
| --- | --- |
| `stand_pde`（属性） | 标准化后的 PDE 表达式 |
| `to_solvable_pde`（属性） | 可提交给 py-pde 的右端项字典 |
| `num_solve(state, t_range, dt, ...)` | 调用 py-pde 进行数值求解 |
| `solve_pde(ctx, state, t_range, dt, ...)` | PDE 适配器函数，`PDEModule.num_solve` 内部调用 |

## 4. 关键参数说明

| 参数 | 含义 |
| --- | --- |
| `sympy_obj` | SymPy 表达式或表达式列表，描述微分方程 |
| `mode` | ODE 求解模式，`'IVP'`（初值问题）或 `'BVP'`（边值问题） |
| `cond` | 初始/边界条件字典，例如 `{y(0): 1}` |
| `const_cond` | 常数/系数取值字典，例如 `{k: 1, m: 2}` |
| `domain` | ODE 自变量采样点，NumPy 一维数组 |
| `solve_method` | ODE 数值方法，如 `'RK45'`、`'RK23'`、`'BDF'` 等 |
| `state` | PDE 初始场，可以是 `py-pde` 的 `Field`、`numpy.ndarray` 或多字段字典 |
| `t_range` | 时间范围，ODE 中常与 `domain` 首尾一致；PDE 中如 `(0, 1)` 或 `1.0` |
| `dt` | PDE 时间步长 |
| `bc` | 边界条件，ODE BVP 中需传入函数；PDE 中可为 `'periodic'` 或 `{'x-': {'value': 0}}` 等 |
| `grid_spec` | PDE 网格说明，例如 `{'bounds': ((0, 1),), 'shape': (64,), 'periodic': True}` |

## 5. 当前限制与缺陷

- **PDE 维度限制**：仅支持最多 2 维空间问题。
- **PDE 混合偏导数**：不支持混合偏导数（如 `∂²u/∂x∂t`），虽然部分混合导数在内部会被拆分为嵌套一阶导数，但稳定性不能保证。
- **PDE 求解速度**：基于 `py-pde` 的显式求解对网格较密或时间步长较小的算例可能较慢。
- **解析解限制**：`ana_solve` 依赖 SymPy `dsolve`，复杂或非线性方程可能无法求出解析解。
- **BVP 支持**：BVP 需要额外提供 `bc` 函数，且初始猜测 `init_guess` 对收敛影响较大。
- **ODE 高阶方程**：系统内部会自动降阶，但用户提供的初始条件需要显式给出各阶导数初值。

## 6. 最简单可运行案例

求解一阶 ODE 初值问题：

```python
import numpy as np
import sympy as sp
from happymath.DiffEq.ODE.ODEModule import ODEModule

y = sp.Function("y")
t = sp.symbols("t")

# dy/dt = 2*y + t, y(0) = 1
ode_expr = -y(t).diff(t, 1) + 2 * y(t) + t
ics = {y(0): 1}

ode_obj = ODEModule(ode_expr)
domain = np.linspace(0, 5, 50)
solution = ode_obj.num_solve(mode="IVP", cond=ics, domain=domain)

print(solution[:5])
```

运行后会得到 `(50, 1)` 的解数组，每行对应 `domain` 中一个时刻的 `y` 值。

## 7. 下一步

- 查看 [examples.md](examples.md) 了解完整案例。
- 查看 [api.md](api.md) 了解每个类/方法的详细参数与返回值。
