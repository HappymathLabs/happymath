"""
Problem definition interfaces.

Standard interface for optimization problems that adapters and solvers can consume.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from sympy import Symbol


class IProblemDefinition(ABC):
    """Interface for optimization problem definitions."""

    @property
    @abstractmethod
    def objective_funcs(self) -> List[Any]:
        """List of objective functions (callables)."""
        pass

    @property
    @abstractmethod
    def objective_exprs(self) -> List[Any]:
        """List of objective expressions (symbolic)."""
        pass

    @property
    @abstractmethod
    def senses(self) -> List[str]:
        """Optimization senses list ('min' or 'max')."""
        pass

    @property
    @abstractmethod
    def parsed_constraints(self) -> List[Any]:
        """List of parsed constraints."""
        pass

    @property
    @abstractmethod
    def all_symbols(self) -> set:
        """Set of all symbols involved."""
        pass

    @property
    @abstractmethod
    def sorted_symbols(self) -> List[Symbol]:
        """Sorted list of decision symbols."""
        pass

    @property
    @abstractmethod
    def variable_bounds(self) -> Tuple[List[float], List[float]]:
        """Variable bounds (lower_bounds, upper_bounds)."""
        pass

    @abstractmethod
    def has_integer_variables(self) -> bool:
        """Whether integer variables exist."""
        pass

    @abstractmethod
    def get_pyomo_problem_type(self) -> str:
        """Return the Pyomo problem type string."""
        pass

    @abstractmethod
    def get_pymoo_problem_type(self) -> Dict[str, Any]:
        """Return a Pymoo problem-type dictionary."""
        pass
