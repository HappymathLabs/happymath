# HappyMath

[![PyPI version](https://badge.fury.io/py/happymath.svg)](https://badge.fury.io/py/happymath)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

HappyMath is a comprehensive mathematical computing and machine learning library that provides unified interfaces for automated machine learning, multi-criteria decision making, differential equations, and mathematical optimization.

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

```bash
pip install happymath
```

**Note**: When installing with pip, the ipopt solver is not included by default, and LightGBM models cannot be properly installed, which may cause AutoML errors. If you install via pip, it is recommended to manually install these two packages through conda.

### Requirements

- Python 3.11+
- All dependencies are automatically installed

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

## Documentation

For detailed documentation, tutorials, and examples, please visit our [documentation page](https://github.com/HappyMathLabs/happymath).

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you have any questions or need help, please:

- Open an issue on GitHub
- Contact us at: tonghui_zou@happymath.com.cn
- Visit our homepage: https://github.com/HappyMathLabs/happymath

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
