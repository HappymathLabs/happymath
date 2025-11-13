"""
HappyMath: A comprehensive mathematical computing and machine learning library.

HappyMath provides a unified interface for:
- Automated Machine Learning (AutoML)
- Multi-Criteria Decision Making (MCDM) 
- Differential Equations (ODE/PDE)
- Mathematical Optimization

Author: HappyMathLabs
Email: tonghui_zou@happymath.com.cn
Homepage: https://github.com/HappyMathLabs/happymath
"""

# Import version from dedicated version module
from ._version import __version__

# Import main modules
from . import AutoML
from . import Decision
from . import DiffEq
from . import Opt

__all__ = [
    "AutoML",
    "Decision", 
    "DiffEq",
    "Opt",
    "__version__",  # Only export version, no other metadata
]