"""
Opt IR module.

Provide unified intermediate representations consumed by parsers and solvers.
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
