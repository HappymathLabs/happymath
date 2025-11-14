# HappyMath

[![PyPI version](https://badge.fury.io/py/happymath.svg)](https://badge.fury.io/py/happymath)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
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
from happymath import AutoML
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')
X, y = data.drop('target', axis=1), data['target']

# Automated classification
automl = AutoML.ClassificationML()
model = automl.fit(X, y)
predictions = model.predict(X_test)
```

### Decision Analysis Example

```python
from happymath import Decision
import numpy as np

# Decision matrix and criteria types
dm_data = np.array([[250, 16, 12], [200, 16, 8], [300, 32, 16]])
criteria = ['min', 'max', 'max']

# Calculate weights and rankings
weighting = Decision.ObjWeighting()
weights = weighting.decide(dataset=dm_data, criterion_type=criteria).get_weights()

scoring = Decision.ScoringDecision()
rankings = scoring.decide(dataset=dm_data, weights=weights, criterion_type=criteria).get_rankings()
print(rankings)
```

### Differential Equations Example

```python
from happymath import DiffEq
import numpy as np

# Define ODE system
def ode_func(t, y):
    return -y + np.sin(t)

# Solve ODE
solver = DiffEq.ODE()
result = solver.solve(ode_func, t_span=[0, 10], y0=[1.0])
t, y = result.get_solution()
```

### Optimization Example

```python
from happymath import Opt
import numpy as np

# Define optimization problem
def objective(x):
    return x[0]**2 + x[1]**2

# Solve optimization problem
optimizer = Opt.Optimization()
result = optimizer.minimize(objective, x0=[1.0, 1.0])
optimal_x = result.x
optimal_value = result.fun
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
