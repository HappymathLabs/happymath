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

__version__ = "0.1.1"
__author__ = "HappyMathLabs"
__email__ = "tonghui_zou@happymath.com.cn"
__description__ = "A comprehensive mathematical computing and machine learning library"
__homepage__ = "https://github.com/HappyMathLabs/happymath"

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
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "__homepage__"
]