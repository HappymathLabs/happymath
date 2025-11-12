"""
Input validation module.

Provides comprehensive validation and clear error messages.
"""

import functools
from typing import Any, Dict, List, Union, Optional
from sympy import Symbol, Basic


def validate_obj_func(obj_func: Any) -> Dict:
    """
    Validate objective function format.

    Args:
        obj_func: Objective function mapping.

    Returns:
        Validated objective mapping.

    Raises:
        TypeError: Wrong type for obj_func
        ValueError: Malformed mapping
    """
    if not isinstance(obj_func, dict):
        raise TypeError(
            f"obj_func must be a dict, got {type(obj_func).__name__}\n"
            f"Expected format: {{'min'/'max': expr}}"
        )

    if len(obj_func) == 0:
        raise ValueError("obj_func mapping cannot be empty")

    # 验证字典的键必须是 'min' 或 'max'
    valid_directions = {'min', 'max'}

    for key, value in obj_func.items():
        # 检查键是否为优化方向
        if isinstance(key, str) and key in valid_directions:
            # Check that value is a valid sympy expression
            if not isinstance(value, (Basic, Symbol)):
                try:
                    from sympy import sympify
                    sympify(str(value))
                except:
                    raise ValueError(
                        f"Objective value must be a valid mathematical expression, got: {value}"
                    )
        else:
            raise ValueError(
                f"Invalid objective mapping.\n"
                f"Expected: {{'min'/'max': expr}}\n"
                f"Got: {{{key}: {value}}}"
            )

    return obj_func


def validate_constraints(constraints: Any) -> Optional[List]:
    """
    Validate constraints format.

    Args:
        constraints: Constraints

    Returns:
        Validated constraint list.

    Raises:
        TypeError: Wrong type
        ValueError: Malformed constraints
    """
    if constraints is None:
        return None

    if not isinstance(constraints, (list, tuple)):
        raise TypeError(
            f"constraints must be a list or tuple, got {type(constraints).__name__}\n"
            f"If a single constraint, pass [constraint]"
        )

    if len(constraints) == 0:
        return []

    # Validate each constraint
    for i, constraint in enumerate(constraints):
        if not isinstance(constraint, Basic):
            try:
                from sympy import sympify
                sympify(str(constraint))
            except:
                raise ValueError(
                    f"constraint[{i}] must be a valid mathematical expression, got: {constraint}"
                )

    return list(constraints)


def validate_mode(mode: Any) -> str:
    """
    Validate solve mode.

    Args:
        mode: Mode string

    Returns:
        Validated mode string

    Raises:
        TypeError: Wrong type
        ValueError: Invalid value
    """
    if not isinstance(mode, str):
        raise TypeError(
            f"mode must be a string, got {type(mode).__name__}"
        )

    valid_modes = {'auto', 'pyomo', 'pymoo'}
    if mode not in valid_modes:
        raise ValueError(
            f"mode must be one of {valid_modes}, got '{mode}'"
        )

    return mode


def validate_search_range(default_search_range: Any) -> Union[int, float]:
    """
    Validate search range.

    Args:
        default_search_range: Default search range

    Returns:
        Validated range value

    Raises:
        TypeError: Wrong type
        ValueError: Invalid value
    """
    if not isinstance(default_search_range, (int, float)):
        raise TypeError(
            f"default_search_range must be numeric, got {type(default_search_range).__name__}"
        )

    if default_search_range <= 0:
        raise ValueError(
            f"default_search_range must be positive, got: {default_search_range}"
        )

    return default_search_range


def validate_tighten_bounds(tighten_bounds: Any) -> Dict[str, Any]:
    """Validate bound tightening configuration."""
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
                f"tighten_bounds must be one of {sorted(allowed_modes)}, got: {tighten_bounds}"
            )
        return {'mode': mode, 'options': {}}

    if isinstance(tighten_bounds, dict):
        if not tighten_bounds:
            return default_config

        options = dict(tighten_bounds)
        mode_value = options.pop('mode', options.pop('strategy', 'auto'))
        nested_options = options.pop('options', None)

        if not isinstance(mode_value, str):
            raise TypeError("tighten_bounds['mode'] must be a string")
        mode = mode_value.strip().lower()
        if mode not in allowed_modes:
            raise ValueError(
                f"tighten_bounds['mode'] must be one of {sorted(allowed_modes)}, got: {mode_value}"
            )

        merged_options = {}
        if isinstance(nested_options, dict):
            merged_options.update(nested_options)
        merged_options.update(options)
        return {'mode': mode, 'options': merged_options}

    raise TypeError(
        "tighten_bounds must be a string, bool or dict, "
        f"got: {type(tighten_bounds).__name__}"
    )


def validate_solver_params(
    solver: Any,
    use_auto_solvers: Any,
    max_solvers: Any
) -> tuple:
    """
    Validate solver parameters.

    Args:
        solver: Solver selection
        use_auto_solvers: Try multiple solvers
        max_solvers: Max number of solvers

    Returns:
        Validated parameter tuple

    Raises:
        TypeError: Wrong types
        ValueError: Invalid values
    """
    # Validate use_auto_solvers
    if not isinstance(use_auto_solvers, bool):
        raise TypeError(
            f"use_auto_solvers must be bool, got {type(use_auto_solvers).__name__}"
        )

    # Validate max_solvers
    if max_solvers != "all":
        if not isinstance(max_solvers, int):
            raise TypeError(
                f"max_solvers must be an int or 'all', got {type(max_solvers).__name__}"
            )
        if max_solvers < 1:
            raise ValueError(
                f"max_solvers must be >= 1, got: {max_solvers}"
            )

    # Validate solver
    if solver is not None:
        if not isinstance(solver, (str, list)):
            raise TypeError(
                f"solver must be None, str, or list[str], got {type(solver).__name__}"
            )

        if isinstance(solver, list):
            if len(solver) == 0:
                raise ValueError("solver list cannot be empty")

            for i, s in enumerate(solver):
                if not isinstance(s, str):
                    raise TypeError(
                        f"solver[{i}] must be a string, got {type(s).__name__}"
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
