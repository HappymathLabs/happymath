"""
优化问题统一中间表示（IR）定义

解析器将 SymPy 表达式转换为 IR，对应的适配器消费 IR 构建求解器模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import sympy as sp


class IROptVarType(str, Enum):
    """变量类型枚举"""

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"
    ENUM = "enum"


@dataclass(slots=True)
class IRDiscreteDomain:
    """离散域信息"""

    values: Tuple[Any, ...]
    labels: Optional[Tuple[str, ...]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple):
            self.values = tuple(self.values)
        if self.labels is not None and not isinstance(self.labels, tuple):
            self.labels = tuple(self.labels)


class IRConstraintCategory(str, Enum):
    """约束类别"""

    ALGEBRAIC = "algebraic"  # 标准代数约束
    DOMAIN = "domain"  # 区间、集合等域约束
    FUNCTIONAL = "functional"  # 积分、微分等功能型约束
    LOGICAL = "logical"  # 条件/分段转换后的逻辑约束


class IRConstraintSense(str, Enum):
    """约束符号"""

    EQ = "eq"
    LE = "le"
    GE = "ge"

@dataclass(slots=True)
class IRConstraint:
    """统一的约束表示"""

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
        """遍历涉及的符号"""
        return self.free_symbols

    def __post_init__(self) -> None:
        if self.free_symbols and not isinstance(self.free_symbols, tuple):
            self.free_symbols = tuple(self.free_symbols)


@dataclass(slots=True)
class IRObjective:
    """目标函数定义"""

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
    """变量定义"""

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
    """优化问题IR顶层结构"""

    variables: List[IROptVariable]
    objectives: List[IRObjective]
    constraints: List[IRConstraint]
    senses: List[str]
    all_symbols: Tuple[sp.Symbol, ...]

    def symbol_to_variable(self) -> Dict[sp.Symbol, IROptVariable]:
        """提供符号到变量定义的映射"""
        return {var.symbol: var for var in self.variables}

    def get_variable(self, symbol: sp.Symbol) -> Optional[IROptVariable]:
        """根据符号获取变量定义"""
        for var in self.variables:
            if var.symbol == symbol:
                return var
        return None

    def __post_init__(self) -> None:
        if self.all_symbols and not isinstance(self.all_symbols, tuple):
            self.all_symbols = tuple(self.all_symbols)
