# HappyMath

[![PyPI version](https://badge.fury.io/py/happymath.svg)](https://badge.fury.io/py/happymath)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

🌐 **Language**: [English](README.md) | [中文](README_zh.md)

---

HappyMath is a comprehensive mathematical computing and machine learning library that provides unified interfaces for automated machine learning, multi-criteria decision making, differential equations, and mathematical optimization.

> ⚠️ **WARNING: PREVIEW VERSION** ⚠️
>
> This is currently a **preview/development version** of HappyMath.
>
> **Please be advised that:**
>
> - This version contains numerous bugs and issues
> - Performance and stability are not guaranteed
> - API may change without notice
> - Documentation may be incomplete or inaccurate
>
> **For production use, please wait for the stable 1.0.0 release.**
>
> We appreciate your interest in testing our library, but use at your own risk!

## Features

### 🤖 AutoML - Automated Machine Learning

- **Classification**: Automated model selection and hyperparameter tuning for classification tasks
- **Regression**: Intelligent regression model building with feature engineering
- **Clustering**: Unsupervised learning with automatic algorithm selection
- **Anomaly Detection**: Outlier and anomaly identification algorithms
- **Time Series**: Specialized time series forecasting and analysis

### 📊 Decision - Multi-Criteria Decision Making (MCDM)

A comprehensive framework for multi-criteria decision analysis with 80+ algorithms:

- **Subjective Weighting**: AHP, BWM, FUCOM, ROC, and more
- **Objective Weighting**: CRITIC, Entropy, MEREC, PSI, and others
- **Scoring Methods**: TOPSIS, VIKOR, SAW, MOORA, and 30+ algorithms
- **Outranking Methods**: ELECTRE and PROMETHEE families
- **Fuzzy Decision Making**: Complete fuzzy methodology support

### 🔧 DiffEq - Differential Equations

Unified interface for solving differential equations:

- **Ordinary Differential Equations (ODE)**: Initial value and boundary value problems
- **Partial Differential Equations (PDE)**: Various numerical methods
- **Symbolic Analysis**: Symbolic computation and analysis tools
- **Multiple Solvers**: SciPy, SymPy, and custom implementations

### ⚙️ Opt - Mathematical Optimization

Comprehensive optimization framework supporting:

- **Linear Programming**: Simplex and interior point methods
- **Nonlinear Programming**: Gradient-based and derivative-free methods
- **Multi-objective Optimization**: Pareto front analysis
- **Constraint Handling**: Various constraint types and formulations
- **Solver Integration**: Pyomo, Pymoo, and specialized solvers

## Installation

### ⭐️ **RECOMMENDED: Conda Installation**

**This is the recommended installation method for optimal compatibility and performance.**

```bash
conda install -c conda-forge happymath
```

### Alternative: Pip Installation

```bash
pip install happymath
```

**⚠️ Important**: When installing with pip, the following issues may occur:

- The ipopt solver is not included by default
- LightGBM models cannot be properly installed
- This may cause AutoML errors and reduced functionality

If you used pip installation or want to ensure all optional dependencies are available, install these packages via conda:

```bash
# Install ipopt solver for optimization problems
conda install -c conda-forge ipopt

# Install LightGBM for enhanced AutoML performance
conda install -c conda-forge lightgbm
```

### Requirements

- Python 3.11+
- All core dependencies are automatically installed

## Quick Start

### AutoML Example

```python
from happymath.AutoML import ClassificationML
from sklearn.datasets import load_iris
import pandas as pd

# Load data
iris = load_iris(as_frame=True)
data = iris.data.copy()
data["target"] = iris.target

# Create a classification experiment
clf = ClassificationML(
    data=data,
    target="target",
    train_size=0.8,
    fold=2,
    seed=42,
    verbose=False,
    html=False,
)

# Train a logistic regression model and predict
model = clf.create("lr", verbose=False)
predictions = clf.predict(data=data.head())
print(predictions[["target", "prediction_label"]].head())
```

### Decision Analysis Example

```python
from happymath.Decision import ObjWeighting, ScoringDecision
import numpy as np

# Decision matrix and criteria types
dm_data = np.array([[250, 16, 12], [200, 16, 8], [300, 32, 16]])
criteria = ["min", "max", "max"]

# Calculate objective weights using entropy
weighting = ObjWeighting(methods=["entropy"])
weights = weighting.decide(
    dataset=dm_data, criterion_type=criteria
).get_weights(method="entropy")
print("Weights:", weights)

# Rank using TOPSIS
scoring = ScoringDecision(methods=["topsis"])
rankings = scoring.decide(
    dataset=dm_data, weights=weights, criterion_type=criteria
).get_rankings(method="topsis")
print("Rankings:", rankings)
```

### Differential Equations Example

```python
import sympy
import numpy as np
from scipy.integrate import solve_ivp
from happymath.DiffEq.ODE.ODEModule import ODEModule

# Define dy/dt = 2*y + t, y(0) = 1
t = sympy.symbols("t")
y = sympy.Function("y")
ode_expr = -y(t).diff(t, 1) + 2 * y(t) + t
ics = {y(0): 1}

ode_obj = ODEModule(ode_expr)
t_span = np.linspace(0, 5, 50)

# Convert to SciPy format and solve
func, y0, const = ode_obj.ode2scipy("IVP", ics)
sol = solve_ivp(func, (0, 5), y0, t_eval=t_span, args=const)
print("y at t=5 ≈", sol.y[0, -1])
```

### Optimization Example

```python
import sympy as sp
from happymath.Opt.OptModule import OptModule

x1, x2 = sp.symbols("x1 x2", real=True)
obj = {"min": (x1 - 1) ** 2 + (x2 - 2) ** 2}
constraints = [x1 >= -5, x1 <= 5, x2 >= -5, x2 <= 5]

opt = OptModule(obj, constraints, mode="pymoo", default_search_range=5.0)
res = opt.solve(solver="GA", use_auto_solvers=False, max_solvers=1)
print("Optimal variables:", res.variables)
print("Optimal value:", res.objective_value)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use HappyMath in your research, please cite:

```bibtex
@software{happymath2024,
  title={HappyMath: A Comprehensive Mathematical Computing Library},
  author={HappyMathLabs},
  year={2024},
  url={https://github.com/HappyMathLabs/happymath}
}
```
