"""
Expression processor.

Provides a unified entrypoint to coordinate analyzers and managers.
"""

from .core.analyzers.objective_analyzer import ObjectiveAnalyzer
from .core.analyzers.constraint_analyzer import ConstraintAnalyzer
from .core.analyzers.problem_type_analyzer import ProblemTypeAnalyzer
from .core.symbol_managers.variable_manager import VariableManager
from .core.symbol_managers.bound_manager import BoundManager
from .core.results.parse_result import ParseResult


class ExpressionProcessor:
    """Unified expression processing entrypoint."""

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
        Process expressions for an optimization problem.

        Args:
            obj_func: Objective dict {"min"/"max": expr}.
            constraints: Optional list of constraints.
            default_search_range: Default search range.
            epsilon: Epsilon used for strict inequalities.
            show_bound_warnings: Whether to warn about variable bounds.
            **kwargs: Extra options.

        Returns:
            ParseResult: Container with analysis results.
        """
        # Read functional config (optional)
        functional_cfg = kwargs.get("functional_config") or kwargs.get("functional_ode")

        obj_analyzer = ObjectiveAnalyzer(obj_func)

        con_analyzer = None
        if constraints is not None:
            con_analyzer = ConstraintAnalyzer(constraints)

        # 2) Symbols/bounds management
        # Extra and excluded symbols (from functional config)
        extra_symbols = []
        exclude_symbols = []
        if functional_cfg is not None:
            try:
                # Support both dataclass and dict
                cfg = functional_cfg
                # Exclude domain variable (e.g., t)
                if hasattr(cfg, "domain") and getattr(cfg.domain, "var", None) is not None:
                    exclude_symbols.append(cfg.domain.var)
                elif isinstance(cfg, dict) and cfg.get("domain") is not None:
                    exclude_symbols.append(cfg["domain"].get("var"))
                # Include control coefficients and extra variables
                if hasattr(cfg, "control") and getattr(cfg.control, "coeff_symbols", None):
                    extra_symbols.extend(list(cfg.control.coeff_symbols))
                if hasattr(cfg, "extra_symbols") and cfg.extra_symbols:
                    extra_symbols.extend(list(cfg.extra_symbols))
                # Parameter symbols (new): param_symbols
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
        # External bounds (from functional config):
        external_bounds = {}
        if functional_cfg is not None:
            try:
                cfg = functional_cfg
                # Unified bounds for control coefficients
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
                # Per-symbol bounds
                per_bounds = getattr(cfg, "bounds", {}) if not isinstance(cfg, dict) else (cfg.get("bounds") or {})
                for s, b in per_bounds.items():
                    try:
                        external_bounds[s] = tuple(b)
                    except Exception:
                        continue
                # Parameter bounds (new): param_bounds
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

        # 3) Problem type analysis
        type_analyzer = ProblemTypeAnalyzer(obj_analyzer, con_analyzer)

        # 4) Wrap results
        return ParseResult(
            obj_analyzer,
            con_analyzer,
            var_manager,
            bound_manager,
            type_analyzer,
            functional_config=functional_cfg,
        )
