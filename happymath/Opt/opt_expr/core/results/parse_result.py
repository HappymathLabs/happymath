"""
Wrapper for parse results.

Encapsulates all parsing and analysis results with standard accessors.
Implements IProblemDefinition to support interface-oriented programming.
"""

import math
from typing import List, Set, Dict, Any, Tuple, Optional
import numpy as np
from sympy import Symbol
from sympy import lambdify
import sympy as sp
from ....interfaces.problem_definition import IProblemDefinition
from ....ir import IRConstraintCategory, IRConstraintSense
from ....functional.spec import FunctionalSpec
from ....functional.evaluator import build_ode_ivp_evaluator
from ....functional.pde_evaluator import build_pde_evaluator
from ....functional.config import ODEIVPConfig, ODEBVPConfig, PDEConfig
from ....ir import (
    IRConstraint,
    IRDiscreteDomain,
    IRObjective,
    IROptProblem,
    IROptVariable,
    IROptVarType,
    IRConstraintCategory,
    IRConstraintSense,
)
from ....adapters.constraint_handlers.epsilon_estimator import EpsilonEstimator


class ParseResult(IProblemDefinition):
    """Container for parsed expression results implementing IProblemDefinition."""

    def __init__(self, obj_analyzer, con_analyzer, var_manager, bound_manager, type_analyzer, functional_config=None):
        """
        Initialize parse result container.

        Args:
            obj_analyzer: ObjectiveAnalyzer instance.
            con_analyzer: ConstraintAnalyzer instance.
            var_manager: VariableManager instance.
            bound_manager: BoundManager instance.
            type_analyzer: ProblemTypeAnalyzer instance.
        """
        self.obj_analyzer = obj_analyzer
        self.con_analyzer = con_analyzer
        self.var_manager = var_manager
        self.bound_manager = bound_manager
        self.type_analyzer = type_analyzer
        self._functional_config = functional_config

        # Ensure analyzers ran (algebraic path)
        self.obj_analyzer.analyze()
        if self.con_analyzer:
            self.con_analyzer.analyze()

        # Build unified IR
        self._ir_problem = self._build_ir_problem()
        self._inject_epsilon_hints()

    # === Internal builders ===

    def _to_float(self, value: Any) -> Optional[float]:
        """Safely convert a value to float."""
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _build_ir_variables(self) -> List[IROptVariable]:
        """Build IR variable list."""
        variables: List[IROptVariable] = []
        sorted_symbols = list(self.var_manager.sorted_symbols)
        xl, xu = self.variable_bounds
        discrete_map = self.bound_manager.discrete_variables if self.bound_manager else {}
        inferred_bounds = self._infer_bounds_from_constraints()

        for idx, symbol in enumerate(sorted_symbols):
            lower = self._to_float(xl[idx]) if xl is not None else None
            upper = self._to_float(xu[idx]) if xu is not None else None
            inferred = inferred_bounds.get(symbol)
            if inferred:
                if inferred.get('lb') is not None:
                    lower = inferred['lb']
                if inferred.get('ub') is not None:
                    upper = inferred['ub']
            if lower is not None and not math.isfinite(lower):
                lower = None
            if upper is not None and not math.isfinite(upper):
                upper = None
            if lower is not None and upper is not None and upper < lower:
                upper = None
            discrete_domain = None
            var_type = IROptVarType.CONTINUOUS

            assumptions = getattr(symbol, "assumptions0", {}) or {}
            assumed_boolean = bool(assumptions.get('boolean'))
            assumed_integer = bool(symbol.is_integer)

            if discrete_map and idx in discrete_map:
                values = tuple(discrete_map[idx])
                unique_values = tuple(dict.fromkeys(values))  # de-dup preserving order
                discrete_domain = IRDiscreteDomain(values=unique_values)
                if set(unique_values).issubset({0, 1}) and len(set(unique_values)) <= 2:
                    var_type = IROptVarType.BINARY
                    lower = 0.0
                    upper = 1.0
                elif all(float(v) == float(int(v)) for v in unique_values):
                    int_values = [int(round(v)) for v in unique_values]
                    int_values_sorted = sorted(set(int_values))
                    if int_values_sorted == list(range(int_values_sorted[0], int_values_sorted[-1] + 1)):
                        var_type = IROptVarType.INTEGER
                        lower = float(int_values_sorted[0])
                        upper = float(int_values_sorted[-1])
                    else:
                        var_type = IROptVarType.ENUM
                        lower = self._to_float(min(unique_values))
                        upper = self._to_float(max(unique_values))
                else:
                    var_type = IROptVarType.ENUM
                    lower = self._to_float(min(unique_values))
                    upper = self._to_float(max(unique_values))

            var_metadata: Dict[str, Any] = {}

            if var_type == IROptVarType.CONTINUOUS:
                if assumed_boolean:
                    var_type = IROptVarType.BINARY
                    lower = 0.0 if lower is None or lower < 0.0 else float(lower)
                    upper = 1.0 if upper is None or upper > 1.0 else float(upper)
                    var_metadata['type_source'] = 'assumption:boolean'
                elif assumed_integer:
                    rounded_lower = None if lower is None else float(math.ceil(lower - 1e-12))
                    rounded_upper = None if upper is None else float(math.floor(upper + 1e-12))
                    if rounded_lower is not None and rounded_upper is not None and rounded_upper < rounded_lower:
                        # 回退到原浮点边界，避免不一致
                        var_type = IROptVarType.INTEGER
                        var_metadata['type_source'] = 'assumption:integer'
                    else:
                        lower = rounded_lower if rounded_lower is not None else lower
                        upper = rounded_upper if rounded_upper is not None else upper
                        var_type = IROptVarType.INTEGER
                        var_metadata['type_source'] = 'assumption:integer'
            elif assumed_boolean and var_type == IROptVarType.INTEGER:
                var_type = IROptVarType.BINARY
                lower = 0.0 if lower is None or lower < 0.0 else float(lower)
                upper = 1.0 if upper is None or upper > 1.0 else float(upper)
                var_metadata['type_source'] = 'assumption:boolean'

            variables.append(IROptVariable(
                symbol=symbol,
                var_type=var_type,
                lower_bound=lower,
                upper_bound=upper,
                discrete_domain=discrete_domain,
                metadata=var_metadata
            ))

        # === 扩展：将 FUNCTIONAL 配置中的参数/控制系数/额外符号注入为决策变量 ===
        # 设计原则：
        # - 仅在符号尚未被 VariableManager 管理时才追加，避免重复；
        # - 边界优先来源：param_bounds / bounds（extra_symbols）/ control.bounds；
        # - 类型默认连续变量；
        try:
            cfg = getattr(self, '_functional_config', None)
            if cfg is not None:
                # 收集待追加的符号（保持声明顺序）
                extra_syms: List[sp.Symbol] = []
                for seq in (
                    list(getattr(cfg, 'param_symbols', []) or []),
                    list(getattr(cfg, 'extra_symbols', []) or []),
                ):
                    for s in seq:
                        if s not in extra_syms:
                            extra_syms.append(s)
                ctrl = getattr(cfg, 'control', None)
                if ctrl is not None and getattr(ctrl, 'coeff_symbols', None):
                    for s in list(ctrl.coeff_symbols):
                        if s not in extra_syms:
                            extra_syms.append(s)

                # 构造快速查询
                param_bounds = dict(getattr(cfg, 'param_bounds', {}) or {})
                extra_bounds = dict(getattr(cfg, 'bounds', {}) or {})
                ctrl_bounds = getattr(ctrl, 'bounds', None) if ctrl is not None else None

                base_set = set(sorted_symbols)
                default_range = float(getattr(self.bound_manager, 'default_search_range', 100.0) or 100.0)

                for s in extra_syms:
                    if s in base_set:
                        continue
                    lb, ub = None, None
                    if s in param_bounds:
                        try:
                            lb, ub = float(param_bounds[s][0]), float(param_bounds[s][1])
                        except Exception:
                            lb, ub = None, None
                    elif s in extra_bounds:
                        try:
                            lb, ub = float(extra_bounds[s][0]), float(extra_bounds[s][1])
                        except Exception:
                            lb, ub = None, None
                    elif ctrl_bounds is not None and s in (getattr(ctrl, 'coeff_symbols', []) or []):
                        try:
                            lb, ub = float(ctrl_bounds[0]), float(ctrl_bounds[1])
                        except Exception:
                            lb, ub = None, None
                    # 兜底：使用默认搜索范围（对称）
                    if lb is None:
                        lb = -default_range
                    if ub is None:
                        ub = default_range

                    variables.append(IROptVariable(
                        symbol=s,
                        var_type=IROptVarType.CONTINUOUS,
                        lower_bound=float(lb),
                        upper_bound=float(ub),
                        discrete_domain=None,
                        metadata={'type_source': 'functional'}
                    ))
                    sorted_symbols.append(s)
        except Exception:
            # 保守：功能型注入失败不影响代数路径
            pass

        return variables

    def _infer_bounds_from_constraints(self) -> Dict[Symbol, Dict[str, Optional[float]]]:
        """从IR约束中推断单变量的显式边界"""
        bounds: Dict[Symbol, Dict[str, Optional[float]]] = {}
        for ir_con in self.parsed_constraints:
            if ir_con.category != IRConstraintCategory.ALGEBRAIC:
                continue
            if ir_con.normalized_expr is None:
                continue
            if ir_con.strict:
                continue
            expr = sp.simplify(ir_con.normalized_expr)
            free_syms = expr.free_symbols
            if len(free_syms) != 1:
                continue
            sym = list(free_syms)[0]
            try:
                solution = sp.solve(sp.Eq(expr, 0), sym)
                if not solution:
                    continue
                value = float(solution[0])
                coeff = expr.diff(sym)
                coeff_val = float(coeff)
            except Exception:
                continue
            if abs(coeff_val) < 1e-12:
                continue
            bound_info = bounds.setdefault(sym, {'lb': None, 'ub': None})
            if ir_con.sense == IRConstraintSense.LE:
                if coeff_val > 0:
                    if bound_info['ub'] is None or value < bound_info['ub']:
                        bound_info['ub'] = value
                else:
                    if bound_info['lb'] is None or value > bound_info['lb']:
                        bound_info['lb'] = value
            elif ir_con.sense == IRConstraintSense.GE:
                if coeff_val > 0:
                    if bound_info['lb'] is None or value > bound_info['lb']:
                        bound_info['lb'] = value
                else:
                    if bound_info['ub'] is None or value < bound_info['ub']:
                        bound_info['ub'] = value
            elif ir_con.sense == IRConstraintSense.EQ:
                bound_info['lb'] = value
                bound_info['ub'] = value
        return bounds

    def _build_ir_objectives(self) -> List[IRObjective]:
        """构建IR目标列表"""
        objectives: List[IRObjective] = []
        parsed_funcs = self.obj_analyzer.parsed_funcs
        senses = self.senses
        obj_exprs = self.objective_exprs

        for idx, expr in enumerate(obj_exprs):
            # Baseline: derive free_symbols and lambda from original expression
            current_expr = expr
            free_symbols = tuple(sorted(list(current_expr.free_symbols), key=lambda s: str(s)))
            lambda_func = parsed_funcs[idx] if idx < len(parsed_funcs) else None

            is_functional = False
            functional_spec = None
            try:
                from sympy import Integral, Derivative
                if getattr(current_expr, 'has', lambda *_: False)(Integral) or \
                   getattr(current_expr, 'has', lambda *_: False)(Derivative):
                    is_functional = True
            except Exception:
                pass

            # If functional config provided, build FUNCTIONAL spec for the objective
            if self._functional_config is not None:
                try:
                    cfg = self._functional_config
                    # 根据配置类型路由 evaluator
                    if isinstance(cfg, PDEConfig):
                        functional_spec = build_pde_evaluator(cfg, objective_key=idx)
                        is_functional = True
                    else:
                        # Default ODE/IVP (compat). BVP evaluator is built by upper layer or future extension.
                        functional_spec = build_ode_ivp_evaluator(cfg, objective_key=idx)
                        is_functional = True
                except Exception:
                    # On failure, fall back to algebraic objective for clearer errors
                    functional_spec = None
                    is_functional = is_functional

            objectives.append(IRObjective(
                sense=senses[idx] if idx < len(senses) else 'min',
                expression=current_expr,
                lambda_func=lambda_func,
                free_symbols=free_symbols,
                is_functional=is_functional,
                functional_spec=functional_spec,
                original=expr,
                metadata={'index': idx}
            ))

        return objectives

    def _build_ir_problem(self) -> IROptProblem:
        """Build the overall IR problem object."""
        variables = self._build_ir_variables()
        objectives = self._build_ir_objectives()
        constraints = list(self.parsed_constraints) if self.parsed_constraints else []

        # Inject virtual FUNCTIONAL constraints from functional config (terminal/path constraints)
        if self._functional_config is not None:
            try:
                meta = getattr(self._functional_config, 'constraint_meta', {}) if not isinstance(self._functional_config, dict) else (self._functional_config.get('constraint_meta') or {})
                for key, cmeta in (meta.items() if isinstance(meta, dict) else []):
                    sense_str = str(cmeta.get('sense', 'eq')).lower()
                    sense = IRConstraintSense.EQ if sense_str == 'eq' else (IRConstraintSense.LE if sense_str == 'le' else IRConstraintSense.GE)
                    spec = build_ode_ivp_evaluator(self._functional_config, objective_key=key, meta_override=cmeta)
                    constraints.append(IRConstraint(
                        identifier=str(key),
                        category=IRConstraintCategory.FUNCTIONAL,
                        sense=sense,
                        functional_spec=spec,
                        original=cmeta,
                        metadata={'aggregation': cmeta.get('aggregation')}
                    ))
            except Exception:
                pass
        senses = list(self.senses)
        # 统一以 IR 变量列表作为 all_symbols，避免功能型注入变量与 manager 不一致
        all_symbols = tuple([v.symbol for v in variables])
        return IROptProblem(
            variables=variables,
            objectives=objectives,
            constraints=constraints,
            senses=senses,
            all_symbols=all_symbols
        )

    def _inject_epsilon_hints(self) -> None:
        """为严格不等式估计 epsilon_hint，提高双端一致性。"""
        if not self.con_analyzer:
            return
        constraints = self.parsed_constraints or []
        if not constraints:
            return
        estimator = EpsilonEstimator()
        symbol_to_index = self.symbol_to_index
        if not symbol_to_index:
            return
        bound_manager = self.bound_manager
        if bound_manager is None:
            return

        try:
            base_lower = np.array(bound_manager.lower_bounds, dtype=float, copy=True)
            base_upper = np.array(bound_manager.upper_bounds, dtype=float, copy=True)
        except Exception:
            return

        default_range = float(getattr(bound_manager, 'default_search_range', 100.0) or 100.0)
        sample_lower = np.where(np.isfinite(base_lower), base_lower, -default_range)
        sample_upper = np.where(np.isfinite(base_upper), base_upper, default_range)
        discrete_vars = bound_manager.discrete_variables or {}

        for ir_con in constraints:
            if not ir_con.strict:
                continue
            if ir_con.lambda_func is None:
                continue
            if not ir_con.free_symbols:
                continue
            try:
                epsilon = estimator.estimate(
                    ir_con.lambda_func,
                    ir_con.free_symbols,
                    symbol_to_index,
                    sample_lower,
                    sample_upper,
                    discrete_vars=discrete_vars,
                    default_epsilon=1e-6
                )
            except Exception:
                continue
            try:
                ir_con.epsilon_hint = float(epsilon)
            except Exception:
                ir_con.epsilon_hint = None

    # === Objectives ===

    @property
    def objective_funcs(self) -> List:
        """Return parsed objective lambda list."""
        return self.obj_analyzer.parsed_funcs

    @property
    def objective_exprs(self) -> List:
        """Return raw objective expressions list."""
        return self.obj_analyzer.obj_func_list

    @property
    def objective_symbols(self) -> List[Set]:
        """Return symbol list per objective."""
        return self.obj_analyzer.symbols_list

    @property
    def senses(self) -> List[str]:
        """Return optimization senses list ('min'/'max')."""
        return self.obj_analyzer.senses

    # === Constraints ===

    # Avoid exposing raw constraints directly (prevent legacy usage)

    @property
    def parsed_constraints(self) -> List:
        """Return parsed constraints list."""
        if self.con_analyzer:
            return self.con_analyzer.parsed_con_list
        return []

    @property
    def discrete_constraints(self) -> List:
        """Return discrete variable constraints."""
        if self.con_analyzer:
            return self.con_analyzer.discrete_constraints
        return []

    @property
    def inequality_constraints(self) -> List:
        """Return inequality constraints."""
        if self.con_analyzer:
            return self.con_analyzer.inequality_constraints
        return []

    @property
    def equality_constraints(self) -> List:
        """Return equality constraints."""
        if self.con_analyzer:
            return self.con_analyzer.equality_constraints
        return []

    # === Variables ===

    @property
    def all_symbols(self) -> Set[Symbol]:
        """Return all symbol variables."""
        return self.var_manager.all_symbols

    @property
    def sorted_symbols(self) -> List[Symbol]:
        """Return sorted symbol list."""
        return self.var_manager.sorted_symbols

    @property
    def symbol_to_index(self) -> Dict[Symbol, int]:
        """Return symbol-to-index mapping."""
        return self.var_manager.symbol_to_index

    @property
    def n_variables(self) -> int:
        """Return number of variables."""
        return self.var_manager.n_variables

    @property
    def variable_bounds(self) -> tuple:
        """Return variable bounds (lower_bounds, upper_bounds)."""
        return self.bound_manager.lower_bounds, self.bound_manager.upper_bounds

    @property
    def discrete_variables(self) -> Dict:
        """Return discrete variable dictionary."""
        return self.bound_manager.discrete_variables

    # === Problem types ===

    @property
    def pyomo_problem_type(self) -> str:
        """Return Pyomo problem type."""
        return self.type_analyzer.pyomo_problem_type

    @property
    def is_convex_qp(self) -> bool:
        """Whether detected as a convex QP."""
        return bool(getattr(self.type_analyzer, 'is_convex_qp', False))

    @property
    def pymoo_problem_type(self) -> Dict[str, Any]:
        """Return Pymoo problem type mapping."""
        return self.type_analyzer.pymoo_problem_type

    def has_integer_variables(self) -> bool:
        """Check if integer/discrete variables exist."""
        if self.con_analyzer:
            return self.con_analyzer.has_integer_variables()
        return False

    # === IProblemDefinition implementation ===

    def get_pyomo_problem_type(self) -> str:
        """Return Pyomo problem type string."""
        return self.pyomo_problem_type

    def get_pymoo_problem_type(self) -> Dict[str, Any]:
        """Return Pymoo problem type dictionary."""
        return self.pymoo_problem_type

    # === IR accessors ===

    @property
    def ir_problem(self) -> IROptProblem:
        """Return the unified IR problem."""
        return self._ir_problem
