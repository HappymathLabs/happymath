"""
通用微分方程（DE）异常体系
精简且适用于 ODE 与 PDE 的统一异常类
"""

from typing import Optional, Any, Dict, List


# ===== 基础异常 =====

class DEException(Exception):
    """DE 模块统一基础异常类"""

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
            base_msg += f" (详情: {detail_str})"
        return base_msg


# ===== 表达式相关 =====

class ExpressionError(DEException):
    """表达式处理异常基类"""
    pass


class InvalidExpressionError(ExpressionError):
    """无效表达式异常"""

    def __init__(self, expression: Any, reason: str = "未知原因"):
        message = f"无效的表达式: {expression}，原因: {reason}"
        super().__init__(message, "EXPR_001", {"expression": str(expression), "reason": reason})


class ExpressionParsingError(ExpressionError):
    """表达式解析异常"""

    def __init__(self, expression: Any, parsing_step: str, original_error: Optional[Exception] = None):
        message = f"表达式解析失败在步骤 '{parsing_step}': {expression}"
        details = {"expression": str(expression), "step": parsing_step}
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(message, "EXPR_002", details)


class ExpressionStandardizationError(ExpressionError):
    """表达式标准化异常"""

    def __init__(self, expression: Any, standardization_type: str, reason: str = ""):
        message = f"表达式标准化失败 ({standardization_type}): {expression}"
        if reason:
            message += f"，原因: {reason}"
        super().__init__(message, "EXPR_003", {
            "expression": str(expression),
            "type": standardization_type,
            "reason": reason
        })


# ===== 验证与条件相关 =====

class ValidationError(DEException):
    """统一验证异常基类"""
    pass


class ConditionError(ValidationError):
    """条件处理异常基类"""
    pass


class ConditionValidationError(ConditionError):
    """条件验证异常"""

    def __init__(self, condition: Any = None, validation_type: str = "", reason: str = ""):
        cond_str = str(condition) if condition is not None else "<unknown>"
        message = f"条件验证失败 ({validation_type}): {cond_str}"
        if reason:
            message += f"，{reason}"
        super().__init__(message, "COND_001", {
            "condition": cond_str,
            "validation_type": validation_type,
            "reason": reason
        })


class FunctionSeparationError(ConditionError):
    """函数分离异常"""

    def __init__(self, function: Any, condition: Any):
        message = f"无法从条件 '{condition}' 中分离函数项 '{function}'"
        super().__init__(message, "COND_002", {
            "function": str(function),
            "condition": str(condition)
        })


class VariableConsistencyError(ConditionError):
    """变量一致性异常"""

    def __init__(self, expected: Any, actual: Any):
        message = f"变量不一致: 期望 '{expected}'，得到 '{actual}'"
        super().__init__(message, "COND_003", {
            "expected": str(expected),
            "actual": str(actual)
        })


class BoundaryConditionError(ConditionError):
    """边界条件异常"""

    def __init__(self, bc_type: str, condition: Any, reason: str = ""):
        message = f"边界条件错误 ({bc_type}): {condition}"
        if reason:
            message += f"，{reason}"
        super().__init__(message, "COND_004", {
            "bc_type": bc_type,
            "condition": str(condition),
            "reason": reason
        })


# ===== 求解器相关 =====

class SolverError(DEException):
    """求解器异常基类"""
    pass


class SolverNotFoundError(SolverError):
    def __init__(self, solver_name: str, available_solvers: Optional[List[str]] = None):
        message = f"未找到求解器: '{solver_name}'"
        details = {"solver_name": solver_name}
        if available_solvers:
            message += f"，可用求解器: {', '.join(available_solvers)}"
            details["available_solvers"] = available_solvers
        super().__init__(message, "SOLV_001", details)


class SolverCreationError(SolverError):
    def __init__(self, solver_name: str, original_error: Optional[Exception] = None):
        message = f"创建求解器失败: '{solver_name}'"
        details = {"solver_name": solver_name}
        if original_error:
            message += f"，原因: {original_error}"
            details["original_error"] = str(original_error)
        super().__init__(message, "SOLV_002", details)


class SolverExecutionError(SolverError):
    def __init__(self, solver_name: str, step: str, original_error: Optional[Exception] = None):
        message = f"求解器执行失败: '{solver_name}' 在步骤 '{step}'"
        details = {"solver_name": solver_name, "step": step}
        if original_error:
            message += f"，原因: {original_error}"
            details["original_error"] = str(original_error)
        super().__init__(message, "SOLV_003", details)


class ConvergenceError(SolverError):
    def __init__(self, solver_name: str, iterations: int, tolerance: float):
        message = f"求解器 '{solver_name}' 未收敛，迭代 {iterations} 次，容差 {tolerance}"
        super().__init__(message, "SOLV_004", {
            "solver_name": solver_name,
            "iterations": iterations,
            "tolerance": tolerance
        })


# ===== 参数相关 =====

class ParameterError(DEException):
    """参数异常基类"""
    pass


class InvalidParameterError(ParameterError):
    def __init__(self, parameter_name: str, value: Any, expected_type: Optional[str] = None,
                 valid_values: Optional[List] = None):
        message = f"无效参数 '{parameter_name}': {value}"
        details = {"parameter": parameter_name, "value": str(value)}
        if expected_type:
            message += f"，期望类型: {expected_type}"
            details["expected_type"] = expected_type
        if valid_values:
            message += f"，有效值: {valid_values}"
            details["valid_values"] = valid_values
        super().__init__(message, "PARAM_001", details)


class MissingParameterError(ParameterError):
    def __init__(self, parameter_name: str, context: str = ""):
        message = f"缺少必需参数: '{parameter_name}'"
        if context:
            message += f" ({context})"
        super().__init__(message, "PARAM_002", {
            "parameter": parameter_name,
            "context": context
        })


class ParameterRangeError(ParameterError):
    def __init__(self, parameter_name: str, value: Any, min_val: Any = None, max_val: Any = None):
        message = f"参数 '{parameter_name}' 值 {value} 超出范围"
        details = {"parameter": parameter_name, "value": str(value)}
        if min_val is not None and max_val is not None:
            message += f" [{min_val}, {max_val}]"
            details["min"] = str(min_val)
            details["max"] = str(max_val)
        elif min_val is not None:
            message += f" (最小值: {min_val})"
            details["min"] = str(min_val)
        elif max_val is not None:
            message += f" (最大值: {max_val})"
            details["max"] = str(max_val)
        super().__init__(message, "PARAM_003", details)


# ===== 兼容别名（保留对旧代码的兼容） =====

ODEBaseException = DEException
ODEValidationError = ValidationError


# ===== 工具函数 =====

def create_detailed_error(exception_class: type, message: str, **kwargs) -> DEException:
    return exception_class(message, **kwargs)


def handle_and_reraise(original_exception: Exception, new_exception_class: type,
                      context: str = "") -> None:
    message = f"操作失败: {str(original_exception)}"
    if context:
        message = f"{context}: {message}"
    raise new_exception_class(message, details={"original_error": str(original_exception)}) from original_exception


def format_error_summary(exceptions: List[Exception]) -> str:
    if not exceptions:
        return "无错误"
    summary = f"发生 {len(exceptions)} 个错误:\n"
    for i, exc in enumerate(exceptions, 1):
        summary += f"{i}. {exc}\n"
    return summary


