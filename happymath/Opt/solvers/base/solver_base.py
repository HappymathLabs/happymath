"""
BaseSolver - common solver base class.

Extracts common logic for PyomoSolver and PymooSolver, removing code duplication.
Provides unified parameter handling and solve flow control.
"""

import time
import warnings
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
from ...interfaces.solver import ISolver
from ...interfaces.problem_definition import IProblemDefinition


class BaseSolver(ISolver):
    """
    Common solver base class.

    Provides unified solver selection and multi-solver management.
    Subclasses implement model creation and single-solver execution.
    """

    def __init__(self, problem: IProblemDefinition):
        """
        Initialize base solver.

        Args:
            problem: Problem definition instance.
        """
        self.problem = problem
        self._model_cache = None

    def solve(
        self,
        solver: Optional[Union[str, List[str]]] = None,
        use_auto_solvers: bool = True,
        max_solvers: Union[int, str] = 3
    ) -> List[Dict[str, Any]]:
        """Unified solve method."""
        start_time = time.time()

        # Parameter validation
        self._validate_solve_parameters(max_solvers)

        # Get model (subclass implements conversion)
        model = self._get_or_create_model()

        # Resolve solver list
        solvers = self._resolve_solver_list(solver, use_auto_solvers, max_solvers)

        # Solve
        results = []
        if len(solvers) == 1:
            # Single solver
            result = self._solve_single(model, solvers[0])
            results.append(result)
        else:
            # Multiple solvers
            results = self._solve_multiple(model, solvers)

        # Record total time
        total_time = time.time() - start_time
        for result in results:
            if 'total_exec_time' not in result:
                result['total_exec_time'] = total_time

        return results

    def _validate_solve_parameters(self, max_solvers: Union[int, str]) -> None:
        """Validate solve parameters."""
        if max_solvers != "all":
            if not isinstance(max_solvers, int):
                raise ValueError("max_solvers must be an int or 'all'")
            if max_solvers < 1:
                raise ValueError("max_solvers must be >= 1")

    def _resolve_solver_list(
        self,
        solver: Optional[Union[str, List[str]]],
        use_auto_solvers: bool,
        max_solvers: Union[int, str]
    ) -> List[str]:
        """Unified solver list resolution logic."""
        if solver is None:
            # Auto-select by problem type when not specified
            default_solvers = self._get_default_solvers(max_solvers)

            if use_auto_solvers:
                return default_solvers
            else:
                return [default_solvers[0]] if default_solvers else []

        elif isinstance(solver, str):
            if use_auto_solvers:
                # Keep specified solver first, then add defaults
                all_solvers = self._get_default_solvers("all")
                remaining = [s for s in all_solvers if s != solver]

                solvers = [solver]
                if max_solvers == "all":
                    solvers.extend(remaining)
                else:
                    solvers.extend(remaining[:max_solvers-1])
                return solvers
            else:
                return [solver]

        elif isinstance(solver, list):
            if use_auto_solvers:
                # User provided multiple solvers
                if max_solvers == "all":
                    return solver
                else:
                    return solver[:max_solvers]
            else:
                return solver

        else:
            raise ValueError("solver must be None, string, or list of strings")

    def _solve_multiple(self, model: Any, solvers: List[str]) -> List[Dict[str, Any]]:
        """Solve with multiple solvers."""
        results = []

        for solver_name in solvers:
            try:
                result = self._solve_single(model, solver_name)
                results.append(result)

                # Optional: continue to compare even after a successful result
                if result.get('success', False):
                    pass  # 继续尝试其他求解器以比较结果

            except Exception as e:
                # Record failure and continue with remaining solvers
                failed_result = {
                    'algorithm': solver_name,
                    'success': False,
                    'message': f"Solver {solver_name} failed: {str(e)}",
                    'solver_type': self.get_solver_type(),
                    'exec_time': 0.0
                }
                results.append(failed_result)

                warnings.warn(f"Solver {solver_name} failed: {str(e)}")

        return results

    # === 抽象方法 - 由子类实现 ===

    @abstractmethod
    def _get_default_solvers(self, max_solvers: Union[int, str]) -> List[str]:
        """
        Get default solver list.

        Args:
            max_solvers: Maximum number of solvers.

        Returns:
            List of default solver names.
        """
        pass

    @abstractmethod
    def _get_or_create_model(self) -> Any:
        """
        Get or create model.

        Returns:
            Framework-specific model.
        """
        pass

    @abstractmethod
    def _solve_single(self, model: Any, solver_name: str) -> Dict[str, Any]:
        """
        Solve with a single solver.

        Args:
            model: Model object
            solver_name: Solver name

        Returns:
            Result dictionary
        """
        pass

    @abstractmethod
    def get_available_solvers(self) -> List[str]:
        """Return list of available solvers."""
        pass

    @abstractmethod
    def get_solver_type(self) -> str:
        """Return solver type."""
        pass
