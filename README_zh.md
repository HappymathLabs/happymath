# HappyMath

[![PyPI版本](https://badge.fury.io/py/happymath.svg)](https://badge.fury.io/py/happymath)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![许可证: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

🌐 **语言**: [English](README.md) | [中文](README_zh.md)

---

HappyMath是一个综合性的数学计算和机器学习库，为自动化机器学习、多准则决策、微分方程和数学优化提供统一接口。

> ⚠️ **警告：预览版本** ⚠️
>
> 这是HappyMath的**预览/开发版本**。
>
> **请注意：**
>
> - 此版本包含许多错误和问题
> - 性能和稳定性无法保证
> - API可能会在没有通知的情况下更改
> - 文档可能不完整或不准确
>
> **生产使用，请等待稳定的1.0.0版本发布。**
>
> 我们感谢您对我们库的测试兴趣，但使用风险自负！

## 功能特性

### 🤖 AutoML - 自动化机器学习

- **分类**：分类任务的自动化模型选择和超参数调优
- **回归**：具有特征工程的智能回归模型构建
- **聚类**：具有自动算法选择的无监督学习
- **异常检测**：异常值和异常识别算法
- **时间序列**：专业的时间序列预测和分析

### 📊 Decision - 多准则决策分析（MCDM）

具有80+算法的多准则决策分析综合框架：

- **主观权重**：AHP、BWM、FUCOM、ROC等
- **客观权重**：CRITIC、熵、MEREC、PSI等
- **评分方法**：TOPSIS、VIKOR、SAW、MOORA等30+算法
- **超越关系方法**：ELECTRE和PROMETHEE系列
- **模糊决策**：完整的模糊方法论支持

### 🔧 DiffEq - 微分方程

求解微分方程的统一接口：

- **常微分方程（ODE）**：初值和边值问题
- **偏微分方程（PDE）**：各种数值方法
- **符号分析**：符号计算和分析工具
- **多种求解器**：SciPy、SymPy和自定义实现

### ⚙️ Opt - 数学优化

支持以下功能的综合优化框架：

- **线性规划**：单纯形和内点法
- **非线性规划**：基于梯度和无导数方法
- **多目标优化**：帕累托前沿分析
- **约束处理**：各种约束类型和公式化
- **求解器集成**：Pyomo、Pymoo和专业求解器

## 安装

### ⭐️ **推荐：Conda安装**

**这是推荐的安装方法，可提供最佳的兼容性和性能。**

```bash
conda install -c conda-forge happymath
```

### 替代方案：Pip安装

```bash
pip install happymath
```

**⚠️ 重要**：使用pip安装时，可能出现以下问题：

- ipopt求解器默认不包含
- LightGBM模型无法正确安装
- 可能导致AutoML错误和功能受限

如果您使用pip安装或希望确保所有可选依赖项都可用，请通过conda安装这些包：

```bash
# 安装用于优化问题的ipopt求解器
conda install -c conda-forge ipopt

# 安装用于增强AutoML性能的LightGBM
conda install -c conda-forge lightgbm
```

### 要求

- Python 3.11+
- 所有核心依赖项都会自动安装

## 快速开始

### AutoML示例

```python
from happymath import AutoML
import pandas as pd

# 加载数据
data = pd.read_csv('your_data.csv')
X, y = data.drop('target', axis=1), data['target']

# 自动化分类
automl = AutoML.ClassificationML()
model = automl.fit(X, y)
predictions = model.predict(X_test)
```

### 决策分析示例

```python
from happymath import Decision
import numpy as np

# 决策矩阵和准则类型
dm_data = np.array([[250, 16, 12], [200, 16, 8], [300, 32, 16]])
criteria = ['min', 'max', 'max']

# 计算权重和排名
weighting = Decision.ObjWeighting()
weights = weighting.decide(dataset=dm_data, criterion_type=criteria).get_weights()

scoring = Decision.ScoringDecision()
rankings = scoring.decide(dataset=dm_data, weights=weights, criterion_type=criteria).get_rankings()
print(rankings)
```

### 微分方程示例

```python
from happymath import DiffEq
import numpy as np

# 定义ODE系统
def ode_func(t, y):
    return -y + np.sin(t)

# 求解ODE
solver = DiffEq.ODE()
result = solver.solve(ode_func, t_span=[0, 10], y0=[1.0])
t, y = result.get_solution()
```

### 优化示例

```python
from happymath import Opt
import numpy as np

# 定义优化问题
def objective(x):
    return x[0]**2 + x[1]**2

# 求解优化问题
optimizer = Opt.Optimization()
result = optimizer.minimize(objective, x0=[1.0, 1.0])
optimal_x = result.x
optimal_value = result.fun
```

## 许可证

本项目基于MIT许可证 - 详见[LICENSE](LICENSE)文件。

## 引用

如果您在研究中使用HappyMath，请引用：

```bibtex
@software{happymath2024,
  title={HappyMath：综合数学计算库},
  author={HappyMathLabs},
  year={2024},
  url={https://github.com/HappyMathLabs/happymath}
}
```

---

[English Version](README.md) | 中文版本
