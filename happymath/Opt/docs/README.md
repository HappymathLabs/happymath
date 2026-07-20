# Opt 文档阅读指南

本目录是 HappyMath Opt 模块（统一数学优化框架）的完整使用说明。编写 Opt 相关代码前，必须先阅读完本目录下所有文档，避免把 Pyomo 求解器、Pymoo 进化算法和 HappyMath 的封装接口混在一起。

## 推荐阅读顺序

1. `README.md`

   先读本文件，了解文档结构、阅读顺序和编码前置要求。

2. `quickstart.md`

   快速建立正确的工作流：理解模块的功能定位、主要类与主要方法、`mode` 参数的三种取值（`"auto"` / `"pyomo"` / `"pymoo"`）以及当前已知限制。这是日常使用 Opt 模块时最先参考的文件。

3. `api.md`

   系统阅读每个类和接口的参数、设计思想、返回形式和 `OptResult` 的结构。尤其要读完 `OptModule.__init__`、`OptModule.solve` 和 `OptResult` 常用属性的说明。

4. `examples.md`

   查看按问题类型组织的完整示例，覆盖 LP、QP、NLP、MILP、多目标优化以及含 ODE 的最优控制等问题。

## 每个文件的含义

| 文件 | 作用 |
|---|---|
| `README.md` | 文档入口和阅读顺序说明。 |
| `quickstart.md` | 面向实际建模的快速决策指南，说明如何选择求解库与求解器。 |
| `api.md` | 接口级参考文档，说明参数、返回值、结果结构和设计思想。 |
| `examples.md` | 可执行示例集合，覆盖不同类型的优化问题。 |

## 编写代码前必须确认

- 已阅读本目录下全部文档，而不是只复制某个示例片段。
- 已明确当前问题类型：LP、QP、NLP、MILP、多目标优化或含 ODE 的最优控制。
- 已确认目标函数格式为 `{"min": expr}` 或 `{"max": expr}`，多目标为 `{"min": [expr1, expr2]}`。
- 已根据问题类型选择 `mode`：单目标有界可用 `"auto"`，多目标只能用 `"pymoo"`。
- 已确认使用 `"pymoo"` 时所有变量都有界；未显式给出上下界的变量会使用 `default_search_range`。
- 已确认多目标问题不能传 `mode="pyomo"`，否则直接报错。
- 已确认所需的外部求解器（如 `cbc`、`glpk`、`ipopt`、`appsi_highs`）在本机可用。
- 已注意当前已知限制：含线性约束的问题建议显式传入 `tighten_bounds="none"`，避免触发 `BoundManager` 内部错误。
- 运行示例或测试时使用 conda 的 `happymath` 环境。
