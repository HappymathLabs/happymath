"""
Opt IR 模块

提供优化问题的统一中间表示定义，供解析器与求解器共同使用。
"""

from .definitions import (
    IRConstraint,
    IRConstraintCategory,
    IRConstraintSense,
    IRDiscreteDomain,
    IRObjective,
    IROptProblem,
    IROptVariable,
    IROptVarType,
)

__all__ = [
    "IRConstraint",
    "IRConstraintCategory",
    "IRConstraintSense",
    "IRDiscreteDomain",
    "IRObjective",
    "IROptProblem",
    "IROptVariable",
    "IROptVarType",
]
