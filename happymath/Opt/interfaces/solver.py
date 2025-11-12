"""
Solver interfaces.

Standard interfaces for optimization solvers supporting various backends.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
from .problem_definition import IProblemDefinition


class ISolver(ABC):
    """Solver interface."""

    @abstractmethod
    def solve(
        self,
        solver: Optional[Union[str, List[str]]] = None,
        use_auto_solvers: bool = True,
        max_solvers: Union[int, str] = 3
    ) -> List[Dict[str, Any]]:
        """
        Solve an optimization problem.

        Args:
            solver: Solver selection
                - None: auto-select by problem type
                - str: use a specific solver
                - list: try multiple solvers
            use_auto_solvers: Whether to try multiple solvers.
            max_solvers: Max number of solvers to try.

        Returns:
            List of solver result dictionaries.
        """
        pass

    @abstractmethod
    def get_available_solvers(self) -> List[str]:
        """Return the list of available solver names."""
        pass

    @abstractmethod
    def get_solver_type(self) -> str:
        """Return solver type ('pyomo' or 'pymoo')."""
        pass


class ISolverFactory(ABC):
    """Solver factory interface."""

    @abstractmethod
    def create_solvers_for(self, problem: IProblemDefinition) -> List[ISolver]:
        """
        Create suitable solvers for a given problem.

        Args:
            problem: Problem definition.

        Returns:
            List of solver instances.
        """
        pass

    @abstractmethod
    def get_supported_problem_types(self) -> List[str]:
        """Return supported problem type list."""
        pass
