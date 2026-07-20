# DiffEq 文档阅读指南

本目录是 HappyMath DiffEq 模块（微分方程求解模块）的完整使用说明。编写 DiffEq 相关代码前，必须先阅读完本目录下所有文档，避免把 ODE 的 SciPy 求解、PDE 的 py-pde 求解和 SymPy 符号表达式描述方式混在一起。

## 推荐阅读顺序

1. `README.md`

   先读本文件，了解文档结构、阅读顺序和编码前置要求。

2. `quickstart.md`

   快速建立正确的工作流：理解 ODE 与 PDE 两条求解路径、入口类与结果容器、关键参数（`mode`、`cond`、`domain`、`state`、`t_range` 等）以及当前限制与缺陷。这是日常使用 DiffEq 模块时最先参考的文件。

3. `api.md`

   系统阅读每个类和接口的参数、设计思想、返回形式和结果结构。尤其要读完 `num_solve()`、`ana_solve()`、`ode2scipy()` 以及 `stand_ode`、`stand_pde`、`to_solvable_pde` 等属性的说明。

4. `examples.md`

   查看按问题类型组织的完整示例，覆盖 ODE 初值问题（IVP）、边值问题（BVP）、方程组、解析解以及基于 py-pde 的 PDE 数值求解。

## 每个文件的含义

| 文件 | 作用 |
|---|---|
| `README.md` | 文档入口和阅读顺序说明。 |
| `quickstart.md` | 面向实际求解的快速指南，说明 ODE/PDE 求解路径和参数选择。 |
| `api.md` | 接口级参考文档，说明参数、返回值、结果结构和设计思想。 |
| `examples.md` | 可执行示例集合，覆盖 ODE 与 PDE 的各类典型问题。 |

## 编写代码前必须确认

- 已阅读本目录下全部文档，而不是只复制某个示例片段。
- 已明确当前问题类型：ODE（一阶/高阶/方程组）还是 PDE，以及是初值问题（IVP）还是边值问题（BVP）。
- 已确认方程用 SymPy 表达式描述（等式右端为零的形式），并正确区分 `cond`（初始/边界条件）与 `const_cond`（常数/系数取值）。
- 已确认 ODE 数值求解时提供了一维 NumPy 数组形式的 `domain`，PDE 求解时提供了 `state`、`t_range`、`dt` 和 `grid_spec`。
- 已确认 PDE 空间维度不超过 2 维，且不依赖混合偏导数（如 `∂²u/∂x∂t`）。
- 已确认 BVP 问题额外提供了 `bc` 函数，并注意初始猜测 `init_guess` 对收敛影响较大。
- 已确认高阶 ODE 的初始条件显式给出了各阶导数初值。
- 已注意 `ana_solve` 依赖 SymPy `dsolve`，复杂或非线性方程可能无解，必要时应改用 `num_solve`。
- 运行示例或测试时使用 conda 的 `happymath` 环境。
