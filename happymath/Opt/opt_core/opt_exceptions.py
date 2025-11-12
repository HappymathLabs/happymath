"""
Optimization exception definitions.

Provide layered exception types with detailed error information.
"""


class OptException(Exception):
    """Base exception for optimization module"""

    def __init__(self, expression=None, operation=None, message=None):
        """
        Initialize the exception.

        Args:
            expression: Expression that caused the error.
            operation: Operation being performed.
            message: Human-readable error message.
        """
        self.expression = expression
        self.operation = operation
        self.message = message or "Optimization error occurred"
        super().__init__(self.message)

    def __str__(self):
        details = [self.message]
        if self.operation:
            details.append(f"Operation: {self.operation}")
        if self.expression:
            details.append(f"Expression: {self.expression}")
        return "\n".join(details)


class InvalidExpressionError(OptException):
    """Invalid expression error"""

    def __init__(self, expression=None, message=None):
        default_message = "Invalid or unsupported expression format"
        super().__init__(
            expression=expression,
            operation="expression validation",
            message=message or default_message
        )


class ConversionError(OptException):
    """Model conversion error"""

    def __init__(self, target_format=None, expression=None, message=None):
        self.target_format = target_format
        default_message = f"Cannot convert to {target_format} format" if target_format else "Model conversion failed"
        super().__init__(
            expression=expression,
            operation=f"convert to {target_format}" if target_format else "model conversion",
            message=message or default_message
        )


class SolverExecutionError(OptException):
    """Solver execution error"""

    def __init__(self, solver_name=None, message=None):
        self.solver_name = solver_name
        default_message = f"Solver {solver_name} failed" if solver_name else "Solver failed"
        super().__init__(
            operation=f"solver execution ({solver_name})" if solver_name else "solver execution",
            message=message or default_message
        )


class ConstraintError(OptException):
    """Constraint handling error"""

    def __init__(self, constraint=None, message=None):
        default_message = "Constraint handling failed"
        super().__init__(
            expression=constraint,
            operation="constraint handling",
            message=message or default_message
        )


class VariableBoundError(OptException):
    """Variable bound error"""

    def __init__(self, variable=None, message=None):
        self.variable = variable
        default_message = f"Invalid bounds for variable {variable}" if variable else "Invalid variable bounds"
        super().__init__(
            expression=variable,
            operation="bound setting",
            message=message or default_message
        )
