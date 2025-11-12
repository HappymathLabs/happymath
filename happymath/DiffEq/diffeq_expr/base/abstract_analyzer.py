"""
Abstract base class for expression analyzers.
Responsibilities:
  - Validate expression correctness
  - Extract metadata such as core functions, symbols, derivatives
  - Determine properties such as linearity
"""

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Set
import sympy


class AbstractExpressionAnalyzer(ABC):
    """Abstract expression analyzer."""
    
    def __init__(self, sympy_obj: Union[sympy.Expr, List], value_range: str = "real"):
        """
        Initialize the analyzer.

        Args:
            sympy_obj: Differential equation expression or list of expressions.
            value_range: Symbol assumptions (e.g., 'real').
        """
        self.sympy_obj = sympy_obj
        self.value_range = value_range
        self._cache: Dict[str, object] = {}

    @abstractmethod
    def is_valid_expression(self) -> bool:
        """Validate expression correctness."""
        pass

    @property
    @abstractmethod
    def expression_type(self) -> str:
        """Return expression type such as 'ODE' or 'PDE'."""
        pass

    @property
    @abstractmethod
    def is_linear(self) -> bool:
        """Whether the expression is linear."""
        pass

    @property
    @abstractmethod
    def core_functions(self) -> List[sympy.Function]:
        """Return list of core functions."""
        pass

    @property
    @abstractmethod
    def core_symbols(self) -> List[sympy.Symbol]:
        """Return list of core symbols."""
        pass

    @property
    @abstractmethod
    def derivative_orders(self) -> Dict[sympy.Derivative, int]:
        """Return mapping from derivative terms to their orders."""
        pass

    @property
    @abstractmethod
    def free_constants(self) -> Set[sympy.Symbol]:
        """Return the set of free constants."""
        pass

    @property
    @abstractmethod
    def expression_order(self) -> int:
        """Return the maximum derivative order in the expression."""
        pass

    @property
    @abstractmethod
    def core_func_symbol_mapping(self) -> Dict:
        """Return mapping between core functions and symbols."""
        pass
    
    def invalidate_cache(self):
        """Invalidate internal caches."""
        self._cache.clear()
    
    def get_cached_value(self, key: str):
        """Get cached value by key."""
        return self._cache.get(key)
    
    def set_cached_value(self, key: str, value):
        """Set cached value by key."""
        self._cache[key] = value
