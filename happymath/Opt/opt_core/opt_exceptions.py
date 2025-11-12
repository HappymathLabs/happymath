"""
优化问题异常类定义

提供分层的异常处理机制，支持详细的错误信息。
"""


class OptException(Exception):
    """优化问题基础异常类"""

    def __init__(self, expression=None, operation=None, message=None):
        """
        初始化异常

        Args:
            expression: 导致异常的表达式
            operation: 执行的操作
            message: 错误消息
        """
        self.expression = expression
        self.operation = operation
        self.message = message or "优化问题发生异常"
        super().__init__(self.message)

    def __str__(self):
        details = [self.message]
        if self.operation:
            details.append(f"操作: {self.operation}")
        if self.expression:
            details.append(f"表达式: {self.expression}")
        return "\n".join(details)


class InvalidExpressionError(OptException):
    """无效表达式异常"""

    def __init__(self, expression=None, message=None):
        default_message = "表达式格式无效或不支持"
        super().__init__(
            expression=expression,
            operation="表达式验证",
            message=message or default_message
        )


class ConversionError(OptException):
    """模型转换异常"""

    def __init__(self, target_format=None, expression=None, message=None):
        self.target_format = target_format
        default_message = f"无法转换为 {target_format} 格式" if target_format else "模型转换失败"
        super().__init__(
            expression=expression,
            operation=f"转换为{target_format}" if target_format else "模型转换",
            message=message or default_message
        )


class SolverExecutionError(OptException):
    """求解器执行异常"""

    def __init__(self, solver_name=None, message=None):
        self.solver_name = solver_name
        default_message = f"求解器 {solver_name} 执行失败" if solver_name else "求解器执行失败"
        super().__init__(
            operation=f"求解器执行({solver_name})" if solver_name else "求解器执行",
            message=message or default_message
        )


class ConstraintError(OptException):
    """约束处理异常"""

    def __init__(self, constraint=None, message=None):
        default_message = "约束处理失败"
        super().__init__(
            expression=constraint,
            operation="约束处理",
            message=message or default_message
        )


class VariableBoundError(OptException):
    """变量边界异常"""

    def __init__(self, variable=None, message=None):
        self.variable = variable
        default_message = f"变量 {variable} 的边界设置无效" if variable else "变量边界设置无效"
        super().__init__(
            expression=variable,
            operation="边界设置",
            message=message or default_message
        )
