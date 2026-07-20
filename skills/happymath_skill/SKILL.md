---
name: happymath-skill
description: "当用户提出以下任务时必须调用本技能：使用 happymath、HappyMath 或 HappyMathLabs/happymath；自动化机器学习、分类、回归、聚类、异常检测、时间序列预测；多准则决策分析、指标赋权、TOPSIS/VIKOR/AHP/熵权法/CRITIC 等决策方法；常微分方程或偏微分方程求解；数学优化、线性规划、非线性规划、多目标优化、最优控制；数学建模竞赛代码实现；科研或工作中的建模求解；安装、配置、更新、排查或验证 happymath 环境；基于 happymath 生成可运行代码、案例代码、课程示例、文档示例或建模流程；数学建模相关内容撰写（如整篇论文或部分章节）。"
---
# HappyMath 使用技能

本技能用于帮助 AI 快速安装、配置、更新、学习和使用 `happymath` 库。若任务涉及数学建模相关的内容撰写（如完整论文或部分章节），请按下方「数学建模写作」部分的指引，访问专用论文写作 Skill。

`happymath` 是一个面向数学建模、应用数学、科研分析和建模竞赛场景的高层 Python 数学建模库。它通过统一封装自动化机器学习、多准则决策、微分方程和数学优化等常见建模方法，降低用户学习和实现数学建模方法的成本。

本技能的核心原则是：

**技能本身只维护稳定的安装、更新、模块索引和动态学习流程，不把具体 API 用法写死，也不把模块范围完全写死。**

当前已知的核心模块包括：

```text
AutoML
Decision
DiffEq
Opt
```

但如果用户任务需要的能力不在这四个模块中，或者 AI 预期某个模块应该存在但本地环境中没有找到，不能直接认定 `happymath` 不支持该任务，而是必须访问项目地址、文档目录和源码目录进行确认。

当 `happymath` 库更新、PyPI 版本更新、API 发生变化、案例文档发生变化、模块新增或模块调整时，AI 必须优先读取线上最新文档和必要源码，再编写代码，而不是依赖本地技能中可能过时的 API 记忆。

---

## 项目地址

优先使用 Gitee 地址：

```text
https://gitee.com/HappymathLabs/happymath
```

当 Gitee 无法访问、访问失败或速度较慢时，使用 GitHub 兜底地址：

```text
https://github.com/HappyMathLabs/happymath
```

访问文档或源码时，优先读取 Gitee；如果失败，再将相同路径映射到 GitHub。

例如：

```text
Gitee:
https://gitee.com/HappymathLabs/happymath/blob/main/happymath/AutoML/docs/README.md

GitHub:
https://github.com/HappyMathLabs/happymath/blob/main/happymath/AutoML/docs/README.md
```

---

## 核心工作原则

### 1. 不依赖本地过时记忆

在编写任何 `happymath` 代码前，必须先确认当前库版本、当前文档和当前 API。

不能因为本技能中写过某个函数、类或参数，就直接假设它一定仍然存在。

### 2. 先判断任务，再选择模块

面对用户任务时，先判断任务属于哪个模块。

当前已知的优先判断规则是：

* 自动化机器学习任务：优先使用 `AutoML`
* 多准则决策分析任务：优先使用 `Decision`
* 微分方程任务：优先使用 `DiffEq`
* 数学优化任务：优先使用 `Opt`

如果任务跨越多个模块，需要先拆解任务，再分别阅读对应模块文档。

如果任务看起来不属于上述四个模块，或者需要一个本技能中未列出的能力，必须先访问项目文档、源码和测试目录，确认 `happymath` 是否已经新增了相关模块或相关接口。

### 3. 先读文档，再写代码

选择模块后，必须读取该模块对应的文档内容，包括：

1. `happymath/<Module>/docs/README.md`（模块文档入口，包含阅读顺序说明）
2. 按 README 指引继续阅读该目录下的 `quickstart.md`、`api.md` 和 `examples.md`
3. 必要时读取对应源码目录
4. 必要时读取 `tests` 目录中的测试样例

只有在完成模块学习后，才能编写代码。

如果找不到对应模块文档，不能直接放弃，需要继续检查：

```text
happymath/<Module>/docs/
happymath/
tests/
```

如果最终确认 `happymath` 中确实没有对应模块或接口，则需要明确告知用户，并使用其他合适的 Python 库或通用方法继续完成任务。

### 4. 所有代码必须在安装了 happymath 的 conda 环境中运行

生成、测试或执行任何 `happymath` 代码时，必须确保代码运行在已经安装 `happymath` 的 conda 环境中。

如果不存在可用环境，必须先创建或配置环境。

### 5. 优先 conda 安装，pip 仅作为兜底

安装 `happymath` 时，必须优先使用 conda：

```bash
conda install -c conda-forge happymath
```

只有在 conda 安装失败、conda-forge 暂无对应版本、依赖解析失败或用户明确要求 pip 时，才允许使用 pip：

```bash
pip install happymath
```

如果使用 pip 安装，仍然必须在 conda 环境中执行 pip，而不是在系统 Python 中执行 pip。

---

## 数学建模写作

当用户任务涉及数学建模相关的内容撰写时，不直接在本 Skill 中实现写作逻辑，而是访问以下专用数学建模论文写作 Skill：

```text
https://gitee.com/HappymathLabs/happymath/blob/main/skills/math-modeling-paper-writer/SKILL.md
https://github.com/HappyMathLabs/happymath/blob/main/skills/math-modeling-paper-writer/SKILL.md
```

触发场景包括但不限于：

- 撰写完整数学建模论文（国赛、美赛、MCM/ICM 等）。
- 撰写论文中的某一部分，如摘要、问题重述、问题分析、模型假设、符号说明、模型建立与求解、模型评价、参考文献、附录等。
- 根据赛题和建模结果生成 solution report。
- 将建模相关 markdown 转换为 docx。

访问该 Skill 后，按其内部流程完成写作任务；本 Skill 不再重复其写作规范。

---

## 环境检查流程

在执行任何 `happymath` 代码前，必须按以下顺序检查环境。

---

### 第一步：检查是否存在 conda

优先使用以下命令：

```bash
conda --version
```

或：

```bash
command -v conda
```

如果命令执行失败，说明当前环境未配置 conda 或 conda 未加入 PATH。

此时需要根据操作系统配置 conda。

---

### 第二步：如果没有 conda，则安装 conda

优先安装 Miniforge 或 Miniconda。

安装前需要判断操作系统：

```bash
uname -s
uname -m
```

常见情况：

* macOS Apple Silicon：`Darwin arm64`
* macOS Intel：`Darwin x86_64`
* Linux x86_64：`Linux x86_64`
* Windows：使用 PowerShell 或图形安装器

安装完成后，必须重新加载 shell 配置或重新打开终端，并再次执行：

```bash
conda --version
```

只有确认 conda 可用后，才能继续。

---

### 第三步：检查已有 conda 环境

执行：

```bash
conda env list
```

或：

```bash
conda info --envs
```

需要检查所有 conda 环境中是否已经安装 `happymath`。

可以用如下方式逐个环境检查：

```bash
conda run -n <env_name> python -c "import importlib.metadata as m; print(m.version('happymath'))"
```

如果某个环境能成功输出版本号，说明该环境已经安装 `happymath`。

---

### 第四步：如果没有任何环境安装 happymath，则创建专用环境

如果所有 conda 环境都没有安装 `happymath`，则创建名为 `happymath` 的新环境：

```bash
conda create -n happymath python=3.11 -y
```

激活环境：

```bash
conda activate happymath
```

安装 `happymath`：

```bash
conda install -c conda-forge happymath -y
```

安装 Jupyter 相关包（如果需要在 Jupyter Notebook / JupyterLab 中使用 `happymath`）：

```bash
conda install -c conda-forge jupyterlab notebook ipykernel -y
```

并将当前环境注册为 Jupyter 内核，方便在 Notebook 中选择：

```bash
python -m ipykernel install --user --name happymath --display-name "Python (happymath)"
```

安装完成后验证：

```bash
python -c "import happymath; print(happymath.__version__ if hasattr(happymath, '__version__') else 'happymath imported successfully')"
```

如果 `happymath.__version__` 不存在，则使用：

```bash
python -c "import importlib.metadata as m; print(m.version('happymath'))"
```

---

## 版本检查与更新流程

如果当前某个 conda 环境已经安装了 `happymath`，必须检查是否为 PyPI 最新版本。

---

### 第一步：获取当前安装版本

在安装了 `happymath` 的 conda 环境中执行：

```bash
python -c "import importlib.metadata as m; print(m.version('happymath'))"
```

---

### 第二步：获取 PyPI 最新版本

执行：

```bash
python - <<'PY'
import json
import urllib.request

url = "https://pypi.org/pypi/happymath/json"
with urllib.request.urlopen(url, timeout=20) as r:
    data = json.load(r)

print(data["info"]["version"])
PY
```

---

### 第三步：比较版本

如果当前安装版本低于 PyPI 最新版本，则需要更新。

优先尝试 conda 更新：

```bash
conda update -c conda-forge happymath -y
```

更新后再次检查版本：

```bash
python -c "import importlib.metadata as m; print(m.version('happymath'))"
```

如果 conda-forge 版本落后于 PyPI，或 conda 更新失败，可以在当前 conda 环境内使用 pip 更新：

```bash
python -m pip install -U happymath
```

更新后再次检查：

```bash
python -c "import importlib.metadata as m; print(m.version('happymath'))"
```

---

## 推荐运行方式

任何使用 `happymath` 的 Python 文件，都应通过 conda 环境运行。

如果环境名为 `happymath`，推荐：

```bash
conda run -n happymath python your_script.py
```

或者先激活环境：

```bash
conda activate happymath
python your_script.py
```

不要在未确认环境的情况下直接运行：

```bash
python your_script.py
```

因为这可能调用系统 Python 或错误的虚拟环境。

---

## 模块总览

`happymath` 当前已知包含四个核心模块：

```text
happymath.AutoML
happymath.Decision
happymath.DiffEq
happymath.Opt
```

这四个模块是当前使用 `happymath` 时的主要入口，但不能将它们理解为永久唯一入口。

如果后续 `happymath` 新增模块、重命名模块、移动模块位置，或用户任务需要本技能未列出的能力，必须通过项目文档、源码和测试样例动态确认当前真实模块结构。

---

## 模块动态确认与兜底流程

在以下情况下，必须执行模块动态确认：

1. 用户任务不明显属于 `AutoML`、`Decision`、`DiffEq`、`Opt` 中的任何一个模块。
2. 用户明确提到一个本技能未列出的 `happymath` 模块。
3. AI 预期某个模块或接口应该存在，但本地环境导入失败。
4. 文档中的模块、源码中的模块和本地安装版本不一致。
5. 用户要求使用某个功能，但四个已知模块都无法覆盖。
6. `happymath` 更新后出现新的文档、新的源码目录或新的测试样例。

动态确认时，必须依次检查：

```text
https://gitee.com/HappymathLabs/happymath/tree/main/happymath
https://gitee.com/HappymathLabs/happymath/tree/main/tests
```

如果 Gitee 无法访问，则切换到：

```text
https://github.com/HappyMathLabs/happymath/tree/main/happymath
https://github.com/HappyMathLabs/happymath/tree/main/tests
```

如果在线项目中存在相关模块或接口，但本地环境中不存在，则优先更新 `happymath`：

```bash
conda update -c conda-forge happymath -y
```

如果 conda 更新后仍不存在，检查 PyPI 最新版本，并在当前 conda 环境中尝试：

```bash
python -m pip install -U happymath
```

如果更新后仍然不存在，需要判断是否出现以下情况：

* 文档超前于已发布版本
* GitHub main 分支代码尚未发布到 PyPI 或 conda-forge
* 本地环境安装源滞后
* 模块已被移除或重命名
* 用户预期的功能尚未实现

如果最终确认 `happymath` 中确实没有对应模块或接口，必须向用户明确说明：

```text
我已经检查了当前安装版本、项目文档、源码目录和测试样例，happymath 当前没有找到能够直接完成该任务的对应模块或接口。
```

随后不能停止任务，而应继续使用其他合适方式完成，例如：

* 使用 Python 标准库
* 使用 NumPy、Pandas、SciPy、Scikit-learn
* 使用 SymPy
* 使用 Pyomo、Pymoo
* 使用 Matplotlib
* 使用其他适合当前任务的建模方法
* 给出不依赖 `happymath` 的可运行实现
* 同时说明如果未来 `happymath` 增加该模块，可以再切换为 `happymath` 实现

不允许因为 `happymath` 没有某个模块就直接放弃用户任务。

---

## 模块一：AutoML

### 模块路径

```python
happymath.AutoML
```

### 功能定位

`AutoML` 模块用于自动化机器学习建模，适合处理常见数据驱动任务。

能力范围包括：

* 分类
* 回归
* 聚类
* 异常检测
* 时间序列预测
* 模型比较
* 模型创建
* 超参数调优
* 模型集成
* 预测
* 模型保存与加载
* 结果表和排行榜输出

### 适用场景

当用户任务中出现以下需求时，优先考虑 `AutoML`：

* 给定数据集，希望自动训练分类模型
* 给定特征和目标列，希望自动回归预测
* 不知道该选什么机器学习模型
* 需要快速比较多个模型效果
* 需要自动调参
* 需要完成聚类或异常检测
* 需要构建时间序列预测基线
* 数学建模竞赛中涉及数据预测、分类、聚类、异常识别等任务

如果 `AutoML` 在本地环境中导入失败，必须先访问项目文档和源码确认该模块是否仍然存在、是否被重命名、是否需要更新版本，而不是直接认定无法使用。

### 文档地址

模块文档入口（README，内含 `quickstart.md`、`api.md`、`examples.md` 的阅读顺序和编码前置要求）：

```text
https://gitee.com/HappymathLabs/happymath/blob/main/happymath/AutoML/docs/README.md
https://github.com/HappyMathLabs/happymath/blob/main/happymath/AutoML/docs/README.md
```

源码目录：

```text
https://gitee.com/HappymathLabs/happymath/tree/main/happymath/AutoML
https://github.com/HappyMathLabs/happymath/tree/main/happymath/AutoML
```

使用 `AutoML` 前，必须先阅读模块文档入口 README，并按其指引读完该目录下全部文档。

---

## 模块二：Decision

### 模块路径

```python
happymath.Decision
```

### 功能定位

`Decision` 模块用于多准则决策分析，也就是 MCDM。

能力范围包括：

* 主观赋权
* 客观赋权
* 综合评价
* 排序评分
* 两两比较决策
* 模糊决策
* 多方法对比
* 权重计算
* 排名输出
* 决策结果解释

常见方法包括但不限于：

* AHP
* BWM
* FUCOM
* ROC
* CRITIC
* Entropy
* MEREC
* PSI
* TOPSIS
* VIKOR
* SAW
* MOORA
* ELECTRE
* PROMETHEE
* 模糊 AHP
* 模糊 TOPSIS
* 模糊 VIKOR

### 适用场景

当用户任务中出现以下需求时，优先考虑 `Decision`：

* 多个方案、多项指标，需要排序
* 需要计算指标权重
* 需要进行综合评价
* 需要进行层次分析法
* 需要用熵权法、CRITIC、TOPSIS、VIKOR 等方法
* 数学建模竞赛中的评价类、决策类、排序类题目
* 科研或工作中需要对多个对象进行综合打分

如果 `Decision` 在本地环境中导入失败，必须先访问项目文档和源码确认该模块是否仍然存在、是否被重命名、是否需要更新版本，而不是直接认定无法使用。

### 文档地址

模块文档入口（README，内含 `quickstart.md`、`api.md`、`examples.md` 的阅读顺序和编码前置要求）：

```text
https://gitee.com/HappymathLabs/happymath/blob/main/happymath/Decision/docs/README.md
https://github.com/HappyMathLabs/happymath/blob/main/happymath/Decision/docs/README.md
```

源码目录：

```text
https://gitee.com/HappymathLabs/happymath/tree/main/happymath/Decision
https://github.com/HappyMathLabs/happymath/tree/main/happymath/Decision
```

使用 `Decision` 前，必须先阅读模块文档入口 README，并按其指引读完该目录下全部文档。

---

## 模块三：DiffEq

### 模块路径

```python
happymath.DiffEq
```

### 功能定位

`DiffEq` 模块用于微分方程建模与求解。

能力范围包括：

* 常微分方程
* ODE 初值问题
* ODE 边值问题
* ODE 符号解析解
* ODE 数值解
* ODE 方程组
* 偏微分方程
* PDE 数值求解
* SymPy 表达式转换
* SciPy 求解器适配

### 适用场景

当用户任务中出现以下需求时，优先考虑 `DiffEq`：

* 建立微分方程模型
* 求解常微分方程
* 求解微分方程组
* 求解析解或数值解
* 求解扩散方程、对流扩散方程等 PDE
* 数学建模竞赛中的动力系统、传播模型、物理过程建模
* 科研或工程中涉及连续系统演化的问题

如果 `DiffEq` 在本地环境中导入失败，必须先访问项目文档和源码确认该模块是否仍然存在、是否被重命名、是否需要更新版本，而不是直接认定无法使用。

### 文档地址

模块文档入口（README，内含 `quickstart.md`、`api.md`、`examples.md` 的阅读顺序和编码前置要求）：

```text
https://gitee.com/HappymathLabs/happymath/blob/main/happymath/DiffEq/docs/README.md
https://github.com/HappyMathLabs/happymath/blob/main/happymath/DiffEq/docs/README.md
```

源码目录：

```text
https://gitee.com/HappymathLabs/happymath/tree/main/happymath/DiffEq
https://github.com/HappyMathLabs/happymath/tree/main/happymath/DiffEq
```

使用 `DiffEq` 前，必须先阅读模块文档入口 README，并按其指引读完该目录下全部文档。

---

## 模块四：Opt

### 模块路径

```python
happymath.Opt
```

### 功能定位

`Opt` 模块用于数学优化建模与求解。

能力范围包括：

* 单目标优化
* 多目标优化
* 线性规划
* 非线性规划
* 约束优化
* 代数优化
* 基于 Pyomo 的数学规划求解
* 基于 Pymoo 的进化优化
* 最优控制
* 微分方程约束优化
* 帕累托前沿分析

### 适用场景

当用户任务中出现以下需求时，优先考虑 `Opt`：

* 需要最大化或最小化目标函数
* 存在约束条件
* 需要求最优参数、最优策略、最优分配
* 需要线性规划或非线性规划
* 需要多目标优化
* 需要遗传算法、NSGA-II 等启发式算法
* 数学建模竞赛中的资源分配、路径规划、调度优化、策略优化
* 科研或工程中的最优化问题

如果 `Opt` 在本地环境中导入失败，必须先访问项目文档和源码确认该模块是否仍然存在、是否被重命名、是否需要更新版本，而不是直接认定无法使用。

### 文档地址

模块文档入口（README，内含 `quickstart.md`、`api.md`、`examples.md` 的阅读顺序和编码前置要求）：

```text
https://gitee.com/HappymathLabs/happymath/blob/main/happymath/Opt/docs/README.md
https://github.com/HappyMathLabs/happymath/blob/main/happymath/Opt/docs/README.md
```

源码目录：

```text
https://gitee.com/HappymathLabs/happymath/tree/main/happymath/Opt
https://github.com/HappyMathLabs/happymath/tree/main/happymath/Opt
```

使用 `Opt` 前，必须先阅读模块文档入口 README，并按其指引读完该目录下全部文档。

---

## 通用文档入口

项目首页：

```text
https://gitee.com/HappymathLabs/happymath
https://github.com/HappyMathLabs/happymath
```

安装文档：

```text
https://gitee.com/HappymathLabs/happymath/blob/main/docs/installation.md
https://github.com/HappyMathLabs/happymath/blob/main/docs/installation.md
```

快速开始：

```text
https://gitee.com/HappymathLabs/happymath/blob/main/docs/quickstart.md
https://github.com/HappyMathLabs/happymath/blob/main/docs/quickstart.md
```

文档首页：

```text
https://gitee.com/HappymathLabs/happymath/blob/main/docs/index.md
https://github.com/HappyMathLabs/happymath/blob/main/docs/index.md
```

各模块文档入口（docs/README.md）所在目录：

```text
https://gitee.com/HappymathLabs/happymath/tree/main/happymath
https://github.com/HappyMathLabs/happymath/tree/main/happymath
```

测试样例目录：

```text
https://gitee.com/HappymathLabs/happymath/tree/main/tests
https://github.com/HappyMathLabs/happymath/tree/main/tests
```

---

## 使用 happymath 解决任务的标准流程

面对用户任务时，必须按以下流程执行。

---

### 第一步：理解用户任务

先判断用户到底要解决什么问题。

需要明确：

* 输入数据是什么
* 输出结果是什么
* 是否需要建模
* 是否需要预测
* 是否需要排序
* 是否需要求解方程
* 是否需要优化
* 是否需要画图
* 是否需要解释结果
* 是否需要给出可复现实验代码

如果用户没有提供完整数据，可以先给出通用代码模板，并明确需要用户补充的数据格式。

---

### 第二步：选择 happymath 模块

根据任务类型优先选择模块：

| 任务类型                                   | 优先模块     |
| ------------------------------------------ | ------------ |
| 分类、回归、聚类、异常检测、时序预测       | `AutoML`   |
| 指标赋权、综合评价、多准则决策、排序       | `Decision` |
| ODE、PDE、动力系统、微分方程求解           | `DiffEq`   |
| 线性规划、非线性规划、多目标优化、最优控制 | `Opt`      |

如果无法确定模块，先阅读 `docs/index.md`、`docs/quickstart.md`、各模块 `happymath/<Module>/docs/README.md` 和源码目录，再判断。

如果四个已知模块都不适合当前任务，必须继续检查项目中是否存在新增模块或其他相关接口。

如果确认没有对应模块，则向用户说明，并使用其他方式继续完成任务。

---

### 第三步：读取模块文档

确定模块后，必须读取该模块 docs 目录下的文档入口 README：

```text
happymath/<Module>/docs/README.md
```

并按 README 中的推荐阅读顺序，继续读完该目录下的：

```text
happymath/<Module>/docs/quickstart.md
happymath/<Module>/docs/api.md
happymath/<Module>/docs/examples.md
```

例如使用 `Decision` 时，需要读取：

```text
happymath/Decision/docs/README.md
```

读取文档时重点关注：

* 当前推荐导入方式
* 当前类名
* 当前函数名
* 当前参数名
* 必填参数
* 返回结果类型
* 结果提取方式
* 文档中的完整案例
* 注意事项
* 依赖限制
* 版本变化提示

---

### 第四步：必要时读取源码和测试

如果文档不足以确定 API，必须继续读取源码和测试样例。

优先读取：

```text
happymath/<Module>/
tests/
```

读取源码时重点关注：

* `__init__.py` 中导出的对象
* 类的构造函数参数
* 方法签名
* 默认参数
* 返回对象
* 异常处理
* 测试文件中的真实调用方式

如果文档和源码不一致，以源码和测试中能运行的方式为准。

---

### 第五步：编写最小可运行代码

在正式实现用户任务前，优先写一个最小可运行代码，用于验证：

* 当前环境可以导入 `happymath`
* 所选模块可以正常导入
* 核心类可以实例化
* 文档中的最小案例可以运行
* 返回结果符合预期

示例验证代码：

```python
import importlib.metadata as metadata
import happymath

print("happymath version:", metadata.version("happymath"))
print("happymath imported successfully")
```

根据模块进一步验证：

```python
from happymath import AutoML
from happymath import Decision
from happymath import DiffEq
from happymath import Opt
```

如果顶层导入方式失败，则必须读取该模块文档和源码，使用当前版本支持的导入方式。

如果某个预期模块导入失败，还必须检查项目文档、源码和测试样例，确认该模块是否被移除、重命名、尚未发布或确实不存在。

---

### 第六步：编写正式代码

正式代码必须满足：

1. 明确导入所需模块
2. 读取或构造用户数据
3. 调用 `happymath` 当前版本支持的 API
4. 输出关键结果
5. 必要时保存结果文件
6. 必要时绘图
7. 必须包含基本异常检查
8. 必须尽量保证用户可以直接运行

代码中不要使用未经文档或源码确认的 API。

如果 `happymath` 中确实没有对应能力，则应给出不依赖 `happymath` 的替代实现，并说明替代原因。

---

### 第七步：运行并验证

所有代码必须在安装 `happymath` 的 conda 环境中运行。

推荐：

```bash
conda run -n happymath python script.py
```

运行后需要检查：

* 是否成功导入
* 是否存在参数错误
* 是否存在依赖错误
* 是否成功得到结果
* 结果形状是否正确
* 结果是否符合任务逻辑

如果失败，需要根据报错回到文档、源码或测试样例中重新确认 API。

---

## 代码生成规范

生成 `happymath` 代码时，应遵循以下规范：

1. 优先使用当前文档中的推荐导入方式。
2. 不使用过时 API。
3. 不虚构类名、函数名和参数。
4. 不直接照搬本技能中的旧示例作为最终依据。
5. 对用户数据格式做必要检查。
6. 如果用户未提供数据，使用可运行的示例数据。
7. 如果任务涉及随机性，设置随机种子。
8. 如果任务需要输出结果，应打印关键结果。
9. 如果任务需要复现，应说明运行环境和版本。
10. 如果代码依赖可选求解器或额外库，应在运行前检查依赖是否存在。
11. 如果本技能列出的四个模块都不适合当前任务，必须检查项目是否已有新增模块。
12. 如果项目中确实没有对应模块，必须明确说明并使用其他方法继续完成任务。

---

## 安装失败处理

如果 conda 安装失败，需要按以下顺序处理：

### 1. 更新 conda

```bash
conda update -n base -c defaults conda -y
```

### 2. 尝试使用 conda-forge

```bash
conda install -c conda-forge happymath -y
```

### 3. 尝试创建干净环境

```bash
conda create -n happymath python=3.11 -y
conda activate happymath
conda install -c conda-forge happymath -y
```

### 4. 如果 conda 仍失败，在 conda 环境内使用 pip

```bash
conda activate happymath
python -m pip install -U happymath
```

### 5. 如果 pip 安装后缺少可选依赖

根据报错补充安装。

常见可选依赖可以优先尝试：

```bash
conda install -c conda-forge ipopt lightgbm -y
```

如果是优化求解器问题，需要根据 `Opt` 文档和报错安装对应求解器。

---

## Gitee 访问失败处理

优先使用 Gitee 访问；如果 Gitee 访问失败，必须自动切换到 GitHub。

路径映射规则：

```text
https://gitee.com/HappymathLabs/happymath/blob/main/<path>
```

替换为：

```text
https://github.com/HappyMathLabs/happymath/blob/main/<path>
```

源码目录映射：

```text
https://gitee.com/HappymathLabs/happymath/tree/main/<path>
```

替换为：

```text
https://github.com/HappyMathLabs/happymath/tree/main/<path>
```

如果需要读取 raw 文件，优先使用 Gitee raw 地址：

```text
https://gitee.com/HappymathLabs/happymath/raw/main/<path>
```

如果 Gitee raw 失败，则使用 GitHub raw 地址尝试读取相同路径：

```text
https://raw.githubusercontent.com/HappyMathLabs/happymath/main/<path>
```

---

## 文档学习要求

使用某个模块前，必须完整阅读该模块相关文档。

不能只读取文档开头，也不能只看一个示例就开始写代码。

对于复杂任务，尤其是 `Decision`、`DiffEq`、`Opt`，必须同时阅读：

* 模块说明文档
* API 文档
* 相关源码
* 相关测试样例

然后再根据用户任务选择最合适的类和方法。

如果任务涉及本技能未列出的模块或能力，需要先完整检查项目文档和源码，再判断是否可以使用 `happymath` 完成。

---

## 任务到模块的判断示例

### 示例一：用户要做分类预测

应优先使用：

```text
AutoML
```

需要阅读：

```text
happymath/AutoML/docs/README.md
```

然后根据 README 指引读完该目录下全部文档，确认分类任务类、参数和预测结果字段。

---

### 示例二：用户要用熵权法 + TOPSIS 综合评价

应优先使用：

```text
Decision
```

需要阅读：

```text
happymath/Decision/docs/README.md
```

然后按 README 指引读完该目录下全部文档，确认客观赋权类、排序决策类、权重提取方式和排名提取方式。

---

### 示例三：用户要解 ODE 或 PDE

应优先使用：

```text
DiffEq
```

需要阅读：

```text
happymath/DiffEq/docs/README.md
```

然后按 README 指引读完该目录下全部文档，确认 ODE 或 PDE 当前推荐调用方式。

---

### 示例四：用户要做线性规划、多目标优化或最优控制

应优先使用：

```text
Opt
```

需要阅读：

```text
happymath/Opt/docs/README.md
```

然后按 README 指引读完该目录下全部文档，确认目标函数、约束、求解模式和结果对象的当前格式。

---

### 示例五：用户要使用本技能未列出的能力

例如用户希望使用 `happymath` 完成一个当前四个模块不明显覆盖的任务。

此时不能直接回答 `happymath` 不支持，而应先检查：

```text
happymath/
happymath/<Module>/docs/
tests/
```

如果发现项目中已经存在相关模块或接口，则阅读对应文档、源码和测试后再使用。

如果确认没有相关模块或接口，则向用户说明，并使用其他方式继续完成任务。

---

## 输出给用户时的要求

当 AI 使用本技能帮助用户时，回答中应包含：

1. 选择了哪个 `happymath` 模块
2. 为什么选择该模块
3. 是否已经确认环境
4. 当前使用的 `happymath` 版本
5. 参考了哪些文档路径
6. 可运行代码
7. 运行方式
8. 结果解释
9. 如果有依赖或版本限制，需要明确说明
10. 如果 `happymath` 没有对应模块或接口，需要明确说明已经检查过项目文档、源码和测试，并给出替代实现方式

如果只是解释思路，不需要执行代码，也应说明推荐模块和应阅读的文档路径。

---

## 不允许的行为

使用本技能时，不允许：

1. 在没有检查环境的情况下直接运行 `happymath` 代码。
2. 在没有读取当前文档的情况下直接假设 API。
3. 把本技能中的旧示例当作最新 API 依据。
4. 在系统 Python 中直接安装或运行 `happymath`。
5. 明明 conda 可用，却优先使用 pip。
6. Gitee 访问失败后直接放弃，不尝试 GitHub。
7. 文档不足时不看源码。
8. 代码报错后不根据报错重新检查文档和源码。
9. 使用未确认存在的函数、类或参数。
10. 忽略用户的具体数据格式和任务目标。
11. 因为本技能只列出了四个核心模块，就认定 `happymath` 永远只有这四个模块。
12. 在预期模块不存在时，不检查项目链接就直接下结论。
13. 在确认 `happymath` 暂不支持某个任务后，直接放弃任务，不提供替代方案。

---

## 最小环境验证脚本

可以创建 `check_happymath_env.py`：

```python
import importlib.metadata as metadata

try:
    import happymath
except Exception as e:
    raise RuntimeError(f"happymath import failed: {e}")

try:
    version = metadata.version("happymath")
except Exception:
    version = "unknown"

print("happymath import success")
print("happymath version:", version)

modules = ["AutoML", "Decision", "DiffEq", "Opt"]

for module_name in modules:
    try:
        module = __import__(f"happymath.{module_name}", fromlist=[module_name])
        print(f"happymath.{module_name} import success")
    except Exception as e:
        print(f"happymath.{module_name} import failed: {e}")
        print(f"需要检查项目文档和源码，确认 {module_name} 是否被移除、重命名、尚未发布或需要更新版本。")
```

运行方式：

```bash
conda run -n happymath python check_happymath_env.py
```

---

## 技能执行总流程摘要

每次使用本技能时，按以下顺序执行：

```text
1. 判断用户任务是否需要 happymath
2. 检查 conda 是否存在
3. 如果没有 conda，则配置 conda
4. 检查所有 conda 环境中是否已有 happymath
5. 如果没有，则创建 happymath 环境并安装
6. 如果已有，则检查是否为 PyPI 最新版本
7. 如果不是最新，则更新
8. 判断任务对应模块
9. 优先在 AutoML、Decision、DiffEq、Opt 中匹配模块
10. 如果四个已知模块不适合，则检查项目是否有新增模块或相关接口
11. 阅读对应模块 happymath/<Module>/docs/README.md
12. 按 README 指引阅读该目录下的 quickstart.md、api.md、examples.md
13. 必要时阅读源码和 tests
14. 如果预期模块不存在，则访问项目链接确认是否真的不存在
15. 如果确认 happymath 没有对应模块或接口，则向用户说明，并使用其他方式继续完成任务
16. 编写最小验证代码
17. 编写正式任务代码
18. 在安装 happymath 的 conda 环境中运行
19. 根据运行结果修正代码
20. 向用户解释模块选择、运行方式和结果
```

---

## 维护原则

本技能应保持稳定，不频繁写入具体 API 细节。

本技能中列出的 `AutoML`、`Decision`、`DiffEq`、`Opt` 是当前已知核心模块，但不是永久固定边界。

当 `happymath` 更新时，不需要同步修改本技能中的具体用法，而是依靠以下机制保证动态同步：

1. 每次使用前检查最新安装版本
2. 每次使用前读取线上最新文档
3. 每次使用前根据任务动态确认模块是否存在
4. 文档不足时读取最新源码
5. API 不确定时运行最小验证代码
6. 以当前环境真实可运行结果为准
7. 如果 `happymath` 没有对应能力，则明确反馈并使用其他合适方法继续完成任务

本技能的价值不在于保存所有 API，也不在于限制 `happymath` 的模块边界，而在于强制 AI 按照正确流程动态学习和使用 `happymath`。
