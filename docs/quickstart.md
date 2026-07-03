# 快速开始

本指南将带你快速了解 HappyMath 四大核心模块的基本用法。每个示例都基于已安装的 `happymath` 库，可直接运行。

## 1. 自动化机器学习（AutoML）

使用 `ClassificationML` 在 Iris 数据集上快速训练一个分类模型：

```python
from happymath.AutoML import ClassificationML
from sklearn.datasets import load_iris
import pandas as pd

# 加载数据
iris = load_iris(as_frame=True)
data = iris.data.copy()
data["target"] = iris.target

# 创建分类实验
clf = ClassificationML(
    data=data,
    target="target",
    train_size=0.8,
    fold=2,
    seed=42,
    verbose=False,
    html=False,
)

# 训练逻辑回归模型并预测
model = clf.create("lr", verbose=False)
predictions = clf.predict(data=data.head())
print(predictions[["target", "prediction_label"]].head())
```

## 2. 多准则决策（Decision）

对决策矩阵计算客观权重并进行 TOPSIS 排序：

```python
from happymath.Decision import ObjWeighting, ScoringDecision
import numpy as np

# 决策矩阵：3 个方案，3 个指标
dm_data = np.array([[250, 16, 12], [200, 16, 8], [300, 32, 16]])
criteria = ["min", "max", "max"]  # 成本型、效益型、效益型

# 计算熵权法权重
weighting = ObjWeighting(methods=["entropy"])
weights = weighting.decide(dataset=dm_data, criterion_type=criteria).get_weights(method="entropy")
print("权重:", weights)

# 使用 TOPSIS 排序
scoring = ScoringDecision(methods=["topsis"])
rankings = scoring.decide(
    dataset=dm_data, weights=weights, criterion_type=criteria
).get_rankings(method="topsis")
print("排序:", rankings)
```

## 3. 常微分方程（DiffEq.ODE）

求解一阶 ODE 初值问题：

```python
import sympy
import numpy as np
from scipy.integrate import solve_ivp
from happymath.DiffEq.ODE.ODEModule import ODEModule

# 定义 dy/dt = 2*y + t, y(0)=1
t = sympy.symbols("t")
y = sympy.Function("y")
ode_expr = -y(t).diff(t, 1) + 2 * y(t) + t
ics = {y(0): 1}

ode_obj = ODEModule(ode_expr)
t_span = np.linspace(0, 5, 50)

# 转换为 SciPy 格式并求解
func, y0, const = ode_obj.ode2scipy("IVP", ics)
sol = solve_ivp(func, (0, 5), y0, t_eval=t_span, args=const)
print("y(5) ≈", sol.y[0, -1])
```

## 4. 数学优化（Opt）

求解一个简单的无约束/箱约束优化问题：

```python
import sympy as sp
from happymath.Opt.OptModule import OptModule

x1, x2 = sp.symbols("x1 x2", real=True)
obj = {"min": (x1 - 1) ** 2 + (x2 - 2) ** 2}
constraints = [x1 >= -5, x1 <= 5, x2 >= -5, x2 <= 5]

opt = OptModule(obj, constraints, mode="pymoo", default_search_range=5.0)
res = opt.solve(solver="GA", use_auto_solvers=False, max_solvers=1)
print("最优解:", res.variables)
print("最优值:", res.objective_value)
```

## 下一步

- 查看 [模块教程](modules/automl.md) 了解各模块的详细用法
- 查看 [API 参考](api/happymath.md) 获取完整的类与方法说明
