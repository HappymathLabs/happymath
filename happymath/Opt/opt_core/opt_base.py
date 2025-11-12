"""
OptBase - 清理后的轻量级协调器

删除了所有向后兼容代码，专注于核心职责：
- 使用ExpressionProcessor处理表达式
- 缓存ParseResult
- 提供问题类型属性（委托给ParseResult）
- 提供模型转换方法（委托给适配器）
"""

from ..opt_expr.processor import ExpressionProcessor

class OptBase:
    """优化问题基础类 - 清理后的轻量级协调器"""

    def __init__(self, obj_func, constraints=None, epsilon=1e-6, default_search_range=100, show_bound_warnings=True, tighten_bounds=None, pyomo_config=None, pymoo_config=None, **kwargs):
        """
        初始化OptBase

        Args:
            obj_func: 目标函数字典 {"min"/"max": expr}
            constraints: 约束条件列表（可选）
            epsilon: epsilon值（用于严格不等式）
            default_search_range: 默认搜索范围
            show_bound_warnings: 是否显示变量边界警告（默认True）
            **kwargs: 其他参数（如仿真优化相关参数）
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
        """获取解析结果"""
        return self._parse_result

    # === 问题类型属性（委托给ParseResult） ===

    @property
    def pyomo_problem_type(self):
        """获取Pyomo问题类型"""
        return self._parse_result.get_pyomo_problem_type()

    @property
    def pymoo_problem_type(self):
        """获取Pymoo问题类型"""
        return self._parse_result.get_pymoo_problem_type()

    def clear_cache(self):
        """清除模型缓存"""
        self._pyomo_model_cache = None
        self._pymoo_problem_cache = None
