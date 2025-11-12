"""
输入验证模块

提供全面的输入验证和清晰的错误处理机制。
"""

import functools
from typing import Any, Dict, List, Union, Optional
from sympy import Symbol, Basic


def validate_obj_func(obj_func: Any) -> Dict:
    """
    验证目标函数格式

    Args:
        obj_func: 目标函数

    Returns:
        验证后的目标函数字典

    Raises:
        TypeError: 如果obj_func类型不正确
        ValueError: 如果obj_func格式不正确
    """
    if not isinstance(obj_func, dict):
        raise TypeError(
            f"目标函数必须是字典类型，当前类型: {type(obj_func).__name__}\n"
            f"正确格式: {{'min'/'max': expr}}"
        )

    if len(obj_func) == 0:
        raise ValueError("目标函数字典不能为空")

    # 验证字典的键必须是 'min' 或 'max'
    valid_directions = {'min', 'max'}

    for key, value in obj_func.items():
        # 检查键是否为优化方向
        if isinstance(key, str) and key in valid_directions:
            # 检查值是否为有效的sympy表达式
            if not isinstance(value, (Basic, Symbol)):
                try:
                    from sympy import sympify
                    sympify(str(value))
                except:
                    raise ValueError(
                        f"目标函数的值必须是有效的数学表达式，当前值: {value}"
                    )
        else:
            raise ValueError(
                f"目标函数格式不正确。\n"
                f"正确格式: {{'min'/'max': expr}}\n"
                f"当前项: {{{key}: {value}}}"
            )

    return obj_func


def validate_constraints(constraints: Any) -> Optional[List]:
    """
    验证约束条件格式

    Args:
        constraints: 约束条件

    Returns:
        验证后的约束条件列表

    Raises:
        TypeError: 如果constraints类型不正确
        ValueError: 如果constraints格式不正确
    """
    if constraints is None:
        return None

    if not isinstance(constraints, (list, tuple)):
        raise TypeError(
            f"约束条件必须是列表或元组类型，当前类型: {type(constraints).__name__}\n"
            f"如果只有一个约束，请使用列表: [constraint]"
        )

    if len(constraints) == 0:
        return []

    # 验证每个约束条件
    for i, constraint in enumerate(constraints):
        if not isinstance(constraint, Basic):
            try:
                from sympy import sympify
                sympify(str(constraint))
            except:
                raise ValueError(
                    f"约束条件[{i}]必须是有效的数学表达式，当前值: {constraint}"
                )

    return list(constraints)


def validate_mode(mode: Any) -> str:
    """
    验证求解模式

    Args:
        mode: 求解模式

    Returns:
        验证后的模式字符串

    Raises:
        TypeError: 如果mode类型不正确
        ValueError: 如果mode值不正确
    """
    if not isinstance(mode, str):
        raise TypeError(
            f"求解模式必须是字符串类型，当前类型: {type(mode).__name__}"
        )

    valid_modes = {'auto', 'pyomo', 'pymoo'}
    if mode not in valid_modes:
        raise ValueError(
            f"求解模式必须是 {valid_modes} 中的一个，当前值: '{mode}'"
        )

    return mode


def validate_search_range(default_search_range: Any) -> Union[int, float]:
    """
    验证搜索范围

    Args:
        default_search_range: 默认搜索范围

    Returns:
        验证后的搜索范围

    Raises:
        TypeError: 如果类型不正确
        ValueError: 如果值不正确
    """
    if not isinstance(default_search_range, (int, float)):
        raise TypeError(
            f"默认搜索范围必须是数值类型，当前类型: {type(default_search_range).__name__}"
        )

    if default_search_range <= 0:
        raise ValueError(
            f"默认搜索范围必须是正数，当前值: {default_search_range}"
        )

    return default_search_range


def validate_tighten_bounds(tighten_bounds: Any) -> Dict[str, Any]:
    """
    验证边界紧化配置
    """
    default_config = {'mode': 'auto', 'options': {}}
    allowed_modes = {'none', 'auto', 'rbc', 'lp'}

    if tighten_bounds is None:
        return default_config

    if isinstance(tighten_bounds, bool):
        return default_config if tighten_bounds else {'mode': 'none', 'options': {}}

    if isinstance(tighten_bounds, str):
        mode = tighten_bounds.strip().lower()
        if mode not in allowed_modes:
            raise ValueError(
                f"tighten_bounds必须是{sorted(allowed_modes)}之一，当前值: {tighten_bounds}"
            )
        return {'mode': mode, 'options': {}}

    if isinstance(tighten_bounds, dict):
        if not tighten_bounds:
            return default_config

        options = dict(tighten_bounds)
        mode_value = options.pop('mode', options.pop('strategy', 'auto'))
        nested_options = options.pop('options', None)

        if not isinstance(mode_value, str):
            raise TypeError("tighten_bounds['mode']必须是字符串")
        mode = mode_value.strip().lower()
        if mode not in allowed_modes:
            raise ValueError(
                f"tighten_bounds['mode']必须是{sorted(allowed_modes)}之一，当前值: {mode_value}"
            )

        merged_options = {}
        if isinstance(nested_options, dict):
            merged_options.update(nested_options)
        merged_options.update(options)
        return {'mode': mode, 'options': merged_options}

    raise TypeError(
        "tighten_bounds必须是字符串、布尔值或字典，"
        f"当前类型: {type(tighten_bounds).__name__}"
    )


def validate_solver_params(
    solver: Any,
    use_auto_solvers: Any,
    max_solvers: Any
) -> tuple:
    """
    验证求解器参数

    Args:
        solver: 求解器设置
        use_auto_solvers: 是否使用多个求解器
        max_solvers: 最大求解器数量

    Returns:
        验证后的参数元组

    Raises:
        TypeError: 如果参数类型不正确
        ValueError: 如果参数值不正确
    """
    # 验证use_auto_solvers
    if not isinstance(use_auto_solvers, bool):
        raise TypeError(
            f"use_auto_solvers必须是布尔类型，当前类型: {type(use_auto_solvers).__name__}"
        )

    # 验证max_solvers
    if max_solvers != "all":
        if not isinstance(max_solvers, int):
            raise TypeError(
                f"max_solvers必须是整数或'all'，当前类型: {type(max_solvers).__name__}"
            )
        if max_solvers < 1:
            raise ValueError(
                f"max_solvers必须大于等于1，当前值: {max_solvers}"
            )

    # 验证solver
    if solver is not None:
        if not isinstance(solver, (str, list)):
            raise TypeError(
                f"solver必须是None、字符串或字符串列表，当前类型: {type(solver).__name__}"
            )

        if isinstance(solver, list):
            if len(solver) == 0:
                raise ValueError("求解器列表不能为空")

            for i, s in enumerate(solver):
                if not isinstance(s, str):
                    raise TypeError(
                        f"求解器列表[{i}]必须是字符串，当前类型: {type(s).__name__}"
                    )

    return solver, use_auto_solvers, max_solvers


def validate_inputs(func):
    """
    输入验证装饰器

    自动验证OptModule.__init__的输入参数
    """
    @functools.wraps(func)
    def wrapper(
        self,
        obj_func,
        constraints=None,
        mode="auto",
        default_search_range=100,
        show_bound_warnings=True,
        tighten_bounds="auto",
        **kwargs
    ):
        try:
            # 验证所有输入参数
            obj_func = validate_obj_func(obj_func)
            constraints = validate_constraints(constraints)
            mode = validate_mode(mode)
            default_search_range = validate_search_range(default_search_range)
            tighten_config = validate_tighten_bounds(tighten_bounds)

            # show_bound_warnings 只需要是布尔类型
            if not isinstance(show_bound_warnings, bool):
                raise TypeError(
                    f"show_bound_warnings必须是布尔类型，当前类型: {type(show_bound_warnings).__name__}"
                )

            # 调用原函数，传递所有参数包括**kwargs
            return func(
                self,
                obj_func,
                constraints,
                mode,
                default_search_range,
                show_bound_warnings,
                tighten_config,
                **kwargs
            )

        except (TypeError, ValueError) as e:
            # 包装错误消息，提供更多上下文
            raise type(e)(
                f"OptModule初始化失败: {str(e)}\n\n"
                f"使用示例:\n"
                f"  from sympy import symbols\n"
                f"  x, y = symbols('x y')\n"
                f"  opt = OptModule({{x**2 + y**2: 'min'}}, [x + y <= 1])"
            ) from e

    return wrapper


def create_detailed_error_message(
    error: Exception,
    context: str,
    suggestions: List[str] = None
) -> str:
    """
    创建详细的错误消息

    Args:
        error: 原始错误
        context: 错误上下文
        suggestions: 修复建议列表

    Returns:
        格式化的错误消息
    """
    message_parts = [
        f"错误位置: {context}",
        f"错误类型: {type(error).__name__}",
        f"错误详情: {str(error)}"
    ]

    if suggestions:
        message_parts.append("修复建议:")
        for i, suggestion in enumerate(suggestions, 1):
            message_parts.append(f"  {i}. {suggestion}")

    return "\n".join(message_parts)
