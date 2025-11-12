"""
Unified intermediate representation (IR) for optimization problems.

Parsers convert SymPy expressions into IR; adapters consume IR to build solver models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import sympy as sp


class IROptVarType(str, Enum):
    """Variable type enumeration."""

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"
    ENUM = "enum"


@dataclass(slots=True)
class IRDiscreteDomain:
    """Discrete domain information."""

    values: Tuple[Any, ...]
    labels: Optional[Tuple[str, ...]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple):
            self.values = tuple(self.values)
        if self.labels is not None and not isinstance(self.labels, tuple):
            self.labels = tuple(self.labels)


class IRConstraintCategory(str, Enum):
    """Constraint category."""

    ALGEBRAIC = "algebraic"  # standard algebraic constraints
    DOMAIN = "domain"        # interval/set domain constraints
    FUNCTIONAL = "functional"# integral/derivative functional constraints
    LOGICAL = "logical"      # logical constraints from conditional/piecewise


class IRConstraintSense(str, Enum):
    """Constraint sense."""

    EQ = "eq"
    LE = "le"
    GE = "ge"

@dataclass(slots=True)
class IRConstraint:
    """Unified constraint representation."""

    identifier: str
    category: IRConstraintCategory
    sense: Optional[IRConstraintSense] = None
    lhs: Optional[sp.Expr] = None
    rhs: Optional[sp.Expr] = None
    normalized_expr: Optional[sp.Expr] = None
    lambda_func: Optional[Callable[..., Any]] = None
    free_symbols: Tuple[sp.Symbol, ...] = field(default_factory=tuple)
    strict: bool = False
    epsilon_hint: Optional[float] = None
    discrete_domain: Optional[IRDiscreteDomain] = None
    functional_spec: Optional[Any] = None
    original: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def iter_symbols(self) -> Iterable[sp.Symbol]:
        """Iterate symbols involved in the constraint."""
        return self.free_symbols

    def __post_init__(self) -> None:
        if self.free_symbols and not isinstance(self.free_symbols, tuple):
            self.free_symbols = tuple(self.free_symbols)


@dataclass(slots=True)
class IRObjective:
    """Objective definition."""

    sense: str
    expression: sp.Expr
    lambda_func: Callable[..., Any]
    free_symbols: Tuple[sp.Symbol, ...]
    is_functional: bool = False
    functional_spec: Optional[Any] = None
    original: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.free_symbols and not isinstance(self.free_symbols, tuple):
            self.free_symbols = tuple(self.free_symbols)


@dataclass(slots=True)
class IROptVariable:
    """Variable definition."""

    symbol: sp.Symbol
    var_type: IROptVarType
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    discrete_domain: Optional[IRDiscreteDomain] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return str(self.symbol)


@dataclass(slots=True)
class IROptProblem:
    """Top-level IR structure for an optimization problem."""

    variables: List[IROptVariable]
    objectives: List[IRObjective]
    constraints: List[IRConstraint]
    senses: List[str]
    all_symbols: Tuple[sp.Symbol, ...]

    def symbol_to_variable(self) -> Dict[sp.Symbol, IROptVariable]:
        """Return mapping from symbol to variable definition."""
        return {var.symbol: var for var in self.variables}

    def get_variable(self, symbol: sp.Symbol) -> Optional[IROptVariable]:
        """Return variable definition by symbol."""
        for var in self.variables:
            if var.symbol == symbol:
                return var
        return None

    def __post_init__(self) -> None:
        if self.all_symbols and not isinstance(self.all_symbols, tuple):
            self.all_symbols = tuple(self.all_symbols)
