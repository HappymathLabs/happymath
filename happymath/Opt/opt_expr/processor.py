"""
表达式处理器

提供统一的表达式处理入口，协调各个组件工作。
"""

from .core.analyzers.objective_analyzer import ObjectiveAnalyzer
from .core.analyzers.constraint_analyzer import ConstraintAnalyzer
from .core.analyzers.problem_type_analyzer import ProblemTypeAnalyzer
from .core.symbol_managers.variable_manager import VariableManager
from .core.symbol_managers.bound_manager import BoundManager
from .core.results.parse_result import ParseResult


class ExpressionProcessor:
    """表达式处理器 - 统一的表达式处理入口"""

    def process(
        self,
        obj_func,
        constraints=None,
        default_search_range=100,
        epsilon=1e-6,
        show_bound_warnings=True,
        tighten_bounds=None,
        **kwargs
    ):
        """
        处理优化问题的表达式

        Args:
            obj_func: 目标函数字典 {"min"/"max": expr}
            constraints: 约束条件列表（可选）
            default_search_range: 默认搜索范围
            epsilon: epsilon值（用于严格不等式）
            show_bound_warnings: 是否显示变量边界警告（默认True）
            **kwargs: 其他参数

        Returns:
            ParseResult: 包含所有解析和分析结果的对象
        """
        # 读取功能型配置（可选）
        functional_cfg = kwargs.get("functional_config") or kwargs.get("functional_ode")

        obj_analyzer = ObjectiveAnalyzer(obj_func)

        con_analyzer = None
        if constraints is not None:
            con_analyzer = ConstraintAnalyzer(constraints)

        # 2) 符号/边界管理
        # 额外符号与排除符号（来自功能型配置）
        extra_symbols = []
        exclude_symbols = []
        if functional_cfg is not None:
            try:
                # 兼容 dataclass 与 dict
                cfg = functional_cfg
                # domain 变量排除（如 t）
                if hasattr(cfg, "domain") and getattr(cfg.domain, "var", None) is not None:
                    exclude_symbols.append(cfg.domain.var)
                elif isinstance(cfg, dict) and cfg.get("domain") is not None:
                    exclude_symbols.append(cfg["domain"].get("var"))
                # 控制系数与额外变量纳入
                if hasattr(cfg, "control") and getattr(cfg.control, "coeff_symbols", None):
                    extra_symbols.extend(list(cfg.control.coeff_symbols))
                if hasattr(cfg, "extra_symbols") and cfg.extra_symbols:
                    extra_symbols.extend(list(cfg.extra_symbols))
                # 参数变量（新增）：param_symbols
                if hasattr(cfg, "param_symbols") and cfg.param_symbols:
                    extra_symbols.extend(list(cfg.param_symbols))
                if isinstance(cfg, dict):
                    ctrl = cfg.get("control") or {}
                    extra_symbols.extend(list(ctrl.get("coeff_symbols") or []))
                    extra_symbols.extend(list(cfg.get("extra_symbols") or []))
                    extra_symbols.extend(list(cfg.get("param_symbols") or []))
            except Exception:
                pass

        var_manager = VariableManager(obj_analyzer, con_analyzer, extra_symbols=extra_symbols, exclude_symbols=exclude_symbols)
        # 外部边界（来自功能型配置）：
        external_bounds = {}
        if functional_cfg is not None:
            try:
                cfg = functional_cfg
                # 控制系数统一边界
                ctrl = getattr(cfg, "control", None) if not isinstance(cfg, dict) else (cfg.get("control") or {})
                coeffs = list(getattr(ctrl, "coeff_symbols", []) if not isinstance(ctrl, dict) else (ctrl.get("coeff_symbols") or []))
                cbounds = None
                if not isinstance(cfg, dict):
                    cbounds = getattr(ctrl, "bounds", None)
                else:
                    cbounds = ctrl.get("bounds")
                if cbounds is not None:
                    for s in coeffs:
                        external_bounds[s] = tuple(cbounds)
                # 逐符号边界
                per_bounds = getattr(cfg, "bounds", {}) if not isinstance(cfg, dict) else (cfg.get("bounds") or {})
                for s, b in per_bounds.items():
                    try:
                        external_bounds[s] = tuple(b)
                    except Exception:
                        continue
                # 参数边界（新增）：param_bounds
                p_bounds = getattr(cfg, "param_bounds", {}) if not isinstance(cfg, dict) else (cfg.get("param_bounds") or {})
                for s, b in (p_bounds.items() if isinstance(p_bounds, dict) else []):
                    try:
                        external_bounds[s] = tuple(b)
                    except Exception:
                        continue
            except Exception:
                pass

        bound_manager = BoundManager(
            var_manager,
            con_analyzer,
            default_search_range,
            show_bound_warnings,
            tighten_config=tighten_bounds,
            external_bounds=external_bounds or None,
        )

        # 3) 问题类型分析
        type_analyzer = ProblemTypeAnalyzer(obj_analyzer, con_analyzer)

        # 4) 封装结果
        return ParseResult(
            obj_analyzer,
            con_analyzer,
            var_manager,
            bound_manager,
            type_analyzer,
            functional_config=functional_cfg,
        )
