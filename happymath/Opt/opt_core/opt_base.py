"""
OptBase - lightweight coordinator after cleanup.

Focus on core responsibilities:
- Use ExpressionProcessor to handle expressions
- Cache ParseResult
- Expose problem-type properties (delegated to ParseResult)
- Provide model conversion via adapters
"""

from ..opt_expr.processor import ExpressionProcessor

class OptBase:
    """Base class for optimization problems - lightweight coordinator."""

    def __init__(self, obj_func, constraints=None, epsilon=1e-6, default_search_range=100, show_bound_warnings=True, tighten_bounds=None, pyomo_config=None, pymoo_config=None, **kwargs):
        """
        Initialize OptBase.

        Args:
            obj_func: Objective dict {"min"/"max": expr}.
            constraints: Optional list of constraints.
            epsilon: Epsilon value for strict inequalities.
            default_search_range: Default search range.
            show_bound_warnings: Whether to warn about variable bounds.
            **kwargs: Extra parameters (e.g., simulation-optimization configs).
        """
        # 使用ExpressionProcessor处理表达式
        processor = ExpressionProcessor()
        self._parse_result = processor.process(
            obj_func,
            constraints,
            default_search_range=default_search_range,
            epsilon=epsilon,
            show_bound_warnings=show_bound_warnings,
            tighten_bounds=tighten_bounds,
            **kwargs
        )

        self.epsilon = epsilon
        self.default_search_range = default_search_range

        # 缓存
        self._pyomo_model_cache = None
        self._pymoo_problem_cache = None

    # === 核心属性访问 ===

    @property
    def parse_result(self):
        """Return the parse result object."""
        return self._parse_result

    # === 问题类型属性（委托给ParseResult） ===

    @property
    def pyomo_problem_type(self):
        """Return Pyomo problem type."""
        return self._parse_result.get_pyomo_problem_type()

    @property
    def pymoo_problem_type(self):
        """Return Pymoo problem-type dictionary."""
        return self._parse_result.get_pymoo_problem_type()

    def clear_cache(self):
        """Clear cached models/problems."""
        self._pyomo_model_cache = None
        self._pymoo_problem_cache = None
