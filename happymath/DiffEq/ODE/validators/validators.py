"""
Validators module providing unified parameter and condition checks.
"""

from collections.abc import Iterable
from typing import Union, List, Any, Optional


class ParameterValidator:
    """统一的参数验证器类"""

    @staticmethod
    def validate_mode(mode: str) -> None:
        if not (mode == "IVP" or mode == "BVP"):
            raise ValueError("The mode parameter passed in is incorrect, please pass in 'IVP' or 'BVP'.")

    @staticmethod
    def validate_domain(domain: Any) -> None:
        if not isinstance(domain, Iterable):
            raise TypeError("The 'domain' parameter only accepts array-like variables, please check your input.")

    @staticmethod
    def validate_init_guess(init_guess: Union[str, Iterable]) -> None:
        if init_guess != "linear" and not isinstance(init_guess, Iterable):
            raise TypeError("The 'init_guess' parameter only accepts 'linear' or 'array-like' variables, please check your input.")

    @staticmethod
    def validate_bvp_boundary_condition(mode: str, bc: Optional[Any]) -> None:
        if mode == "BVP" and bc is None:
            raise ValueError("For BVP the 'bc' function must be passed in.")

    @staticmethod
    def validate_condition_dict(cond: Any) -> None:
        if not isinstance(cond, dict):
            raise TypeError("Parameter 'cond' must be a dictionary.")

    @staticmethod
    def validate_const_condition_dict(const_cond: Any) -> None:
        if const_cond is not None and not isinstance(const_cond, dict):
            raise TypeError("Parameter 'const_cond' only accept 'dict' type!")

    @staticmethod
    def validate_tolerance(tol: float, param_name: str = "tol") -> None:
        if not isinstance(tol, (int, float)) or tol <= 0:
            raise ValueError(f"Parameter '{param_name}' must be a positive number.")

    @staticmethod
    def validate_solver_parameters(mode: str, domain: Any, init_guess: Union[str, Iterable], bc: Optional[Any] = None) -> None:
        ParameterValidator.validate_mode(mode)
        ParameterValidator.validate_domain(domain)
        ParameterValidator.validate_init_guess(init_guess)
        ParameterValidator.validate_bvp_boundary_condition(mode, bc)

    @staticmethod
    def validate_numeric_value(value: Any, param_name: str, allow_negative: bool = True) -> None:
        if not isinstance(value, (int, float, complex)):
            raise TypeError(f"Parameter '{param_name}' must be a number.")
        if not allow_negative and value < 0:
            raise ValueError(f"Parameter '{param_name}' must be non-negative.")

    @staticmethod
    def validate_iterable_not_empty(iterable: Any, param_name: str) -> None:
        if not isinstance(iterable, Iterable):
            raise TypeError(f"Parameter '{param_name}' must be iterable.")
        if len(list(iterable)) == 0:
            raise ValueError(f"Parameter '{param_name}' cannot be empty.")


class ODEValidationError(Exception):
    pass


class ConditionValidationError(ODEValidationError):
    pass


class FunctionSeparationError(ConditionValidationError):
    pass


class VariableConsistencyError(ConditionValidationError):
    pass


class ExpressionSeparationError(ConditionValidationError):
    pass


class BoundaryConditionError(ODEValidationError):
    pass


class BVPValidationError(BoundaryConditionError):
    pass


class SolverConfigurationError(ODEValidationError):
    pass


class ErrorCalculationError(ODEValidationError):
    pass


class CacheError(ODEValidationError):
    pass

__all__ = [
    "ParameterValidator",
    "ODEValidationError",
    "ConditionValidationError",
    "FunctionSeparationError",
    "VariableConsistencyError",
    "ExpressionSeparationError",
    "BoundaryConditionError",
    "BVPValidationError",
    "SolverConfigurationError",
    "ErrorCalculationError",
    "CacheError",
]

