"""
Unified exception hierarchy for differential equations (DE).
Compact, shared across ODE and PDE modules.
"""

from typing import Optional, Any, Dict, List


# ===== Base Exceptions =====

class DEException(Exception):
    """Base exception for DE modules"""

    def __init__(self, message: str, error_code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self) -> str:
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.details:
            detail_str = ", ".join([f"{k}: {v}" for k, v in self.details.items()])
            base_msg += f" (details: {detail_str})"
        return base_msg


# ===== Expression Related =====

class ExpressionError(DEException):
    """Base class for expression-related errors"""
    pass


class InvalidExpressionError(ExpressionError):
    """Invalid expression error"""

    def __init__(self, expression: Any, reason: str = "未知原因"):
        message = f"Invalid expression: {expression}, reason: {reason}"
        super().__init__(message, "EXPR_001", {"expression": str(expression), "reason": reason})


class ExpressionParsingError(ExpressionError):
    """Expression parsing error"""

    def __init__(self, expression: Any, parsing_step: str, original_error: Optional[Exception] = None):
        message = f"Failed to parse expression at step '{parsing_step}': {expression}"
        details = {"expression": str(expression), "step": parsing_step}
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(message, "EXPR_002", details)


class ExpressionStandardizationError(ExpressionError):
    """Expression standardization error"""

    def __init__(self, expression: Any, standardization_type: str, reason: str = ""):
        message = f"Expression standardization failed ({standardization_type}): {expression}"
        if reason:
            message += f", reason: {reason}"
        super().__init__(message, "EXPR_003", {
            "expression": str(expression),
            "type": standardization_type,
            "reason": reason
        })


# ===== Validation and Conditions =====

class ValidationError(DEException):
    """Base class for validation errors"""
    pass


class ConditionError(ValidationError):
    """Base class for condition errors"""
    pass


class ConditionValidationError(ConditionError):
    """Condition validation error"""

    def __init__(self, condition: Any = None, validation_type: str = "", reason: str = ""):
        cond_str = str(condition) if condition is not None else "<unknown>"
        message = f"Condition validation failed ({validation_type}): {cond_str}"
        if reason:
            message += f", {reason}"
        super().__init__(message, "COND_001", {
            "condition": cond_str,
            "validation_type": validation_type,
            "reason": reason
        })


class FunctionSeparationError(ConditionError):
    """Function separation error"""

    def __init__(self, function: Any, condition: Any):
        message = f"Unable to separate function '{function}' from condition '{condition}'"
        super().__init__(message, "COND_002", {
            "function": str(function),
            "condition": str(condition)
        })


class VariableConsistencyError(ConditionError):
    """Variable consistency error"""

    def __init__(self, expected: Any, actual: Any):
        message = f"Variable inconsistency: expected '{expected}', got '{actual}'"
        super().__init__(message, "COND_003", {
            "expected": str(expected),
            "actual": str(actual)
        })


class BoundaryConditionError(ConditionError):
    """Boundary condition error"""

    def __init__(self, bc_type: str, condition: Any, reason: str = ""):
        message = f"Boundary condition error ({bc_type}): {condition}"
        if reason:
            message += f", {reason}"
        super().__init__(message, "COND_004", {
            "bc_type": bc_type,
            "condition": str(condition),
            "reason": reason
        })


# ===== Solver Related =====

class SolverError(DEException):
    """Base class for solver errors"""
    pass


class SolverNotFoundError(SolverError):
    def __init__(self, solver_name: str, available_solvers: Optional[List[str]] = None):
        message = f"Solver not found: '{solver_name}'"
        details = {"solver_name": solver_name}
        if available_solvers:
            message += f", available solvers: {', '.join(available_solvers)}"
            details["available_solvers"] = available_solvers
        super().__init__(message, "SOLV_001", details)


class SolverCreationError(SolverError):
    def __init__(self, solver_name: str, original_error: Optional[Exception] = None):
        message = f"Failed to create solver: '{solver_name}'"
        details = {"solver_name": solver_name}
        if original_error:
            message += f", reason: {original_error}"
            details["original_error"] = str(original_error)
        super().__init__(message, "SOLV_002", details)


class SolverExecutionError(SolverError):
    def __init__(self, solver_name: str, step: str, original_error: Optional[Exception] = None):
        message = f"Solver execution failed: '{solver_name}' at step '{step}'"
        details = {"solver_name": solver_name, "step": step}
        if original_error:
            message += f", reason: {original_error}"
            details["original_error"] = str(original_error)
        super().__init__(message, "SOLV_003", details)


class ConvergenceError(SolverError):
    def __init__(self, solver_name: str, iterations: int, tolerance: float):
        message = f"Solver '{solver_name}' did not converge, iterations {iterations}, tolerance {tolerance}"
        super().__init__(message, "SOLV_004", {
            "solver_name": solver_name,
            "iterations": iterations,
            "tolerance": tolerance
        })


# ===== Parameter Related =====

class ParameterError(DEException):
    """Base class for parameter errors"""
    pass


class InvalidParameterError(ParameterError):
    def __init__(self, parameter_name: str, value: Any, expected_type: Optional[str] = None,
                 valid_values: Optional[List] = None):
        message = f"Invalid parameter '{parameter_name}': {value}"
        details = {"parameter": parameter_name, "value": str(value)}
        if expected_type:
            message += f", expected type: {expected_type}"
            details["expected_type"] = expected_type
        if valid_values:
            message += f", valid values: {valid_values}"
            details["valid_values"] = valid_values
        super().__init__(message, "PARAM_001", details)


class MissingParameterError(ParameterError):
    def __init__(self, parameter_name: str, context: str = ""):
        message = f"Missing required parameter: '{parameter_name}'"
        if context:
            message += f" ({context})"
        super().__init__(message, "PARAM_002", {
            "parameter": parameter_name,
            "context": context
        })


class ParameterRangeError(ParameterError):
    def __init__(self, parameter_name: str, value: Any, min_val: Any = None, max_val: Any = None):
        message = f"Parameter '{parameter_name}' value {value} out of range"
        details = {"parameter": parameter_name, "value": str(value)}
        if min_val is not None and max_val is not None:
            message += f" [{min_val}, {max_val}]"
            details["min"] = str(min_val)
            details["max"] = str(max_val)
        elif min_val is not None:
            message += f" (min: {min_val})"
            details["min"] = str(min_val)
        elif max_val is not None:
            message += f" (max: {max_val})"
            details["max"] = str(max_val)
        super().__init__(message, "PARAM_003", details)


# ===== Compatibility Aliases (legacy support) =====

ODEBaseException = DEException
ODEValidationError = ValidationError


# ===== Utilities =====

def create_detailed_error(exception_class: type, message: str, **kwargs) -> DEException:
    return exception_class(message, **kwargs)


def handle_and_reraise(original_exception: Exception, new_exception_class: type,
                      context: str = "") -> None:
    """Wrap and re-raise an exception with additional context.

    Args:
        original_exception: The caught exception to wrap.
        new_exception_class: Exception class to raise.
        context: Optional short description of the operation.
    """
    message = f"Operation failed: {str(original_exception)}"
    if context:
        message = f"{context}: {message}"
    raise new_exception_class(message, details={"original_error": str(original_exception)}) from original_exception


def format_error_summary(exceptions: List[Exception]) -> str:
    """Format a human-readable summary for a list of exceptions."""
    if not exceptions:
        return "No errors"
    summary = f"Occurred {len(exceptions)} errors:\n"
    for i, exc in enumerate(exceptions, 1):
        summary += f"{i}. {exc}\n"
    return summary

