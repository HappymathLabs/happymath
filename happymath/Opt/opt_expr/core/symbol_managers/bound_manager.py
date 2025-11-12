"""
Bound manager.

Manages variable bounds:
- Extract simple bound constraints
- Apply default search ranges
- Handle discrete variable domains
"""

import numpy as np
import sympy as sp
import warnings
from typing import Dict, Set, Optional, List, Tuple, Any
from sympy import Symbol
from ....ir import IRConstraintSense


class BoundManager:
    """Manager for variable bounds and discrete domains."""

    def __init__(self, var_manager, con_analyzer=None, default_search_range=100, show_bound_warnings=True, tighten_config=None, external_bounds: Dict[Symbol, tuple] | None = None):
        """
        Initialize the bound manager.

        Args:
            var_manager: VariableManager instance.
            con_analyzer: Optional ConstraintAnalyzer instance.
            default_search_range: Default search range.
            show_bound_warnings: Whether to warn for variable bounds.
        """
        self.var_manager = var_manager
        self.con_analyzer = con_analyzer
        self.default_search_range = default_search_range
        self.show_bound_warnings = show_bound_warnings
        if isinstance(tighten_config, dict):
            tighten_payload = tighten_config
        elif tighten_config is None:
            tighten_payload = {}
        else:
            tighten_payload = {'mode': tighten_config}
        mode = str(tighten_payload.get('mode', 'auto')).lower()
        if mode not in {'none', 'auto', 'rbc', 'lp'}:
            mode = 'auto'
        self._tighten_strategy = 'rbc' if mode == 'auto' else mode
        self._tighten_options = dict(tighten_payload.get('options') or {})

        self._true_lower_bounds = None
        self._true_upper_bounds = None
        self._discrete_variables = None  # {var_idx: [values]}
        self._linear_rows_cache = None
        self._auto_tightening_done = False
        self._coeff_tol = 1e-12

    def extract_bounds(self) -> tuple:
        """
        Extract variable bounds.

        Returns:
            tuple of two numpy arrays: (lower_bounds, upper_bounds)
        """
        self._ensure_true_bounds()
        return self._true_lower_bounds, self._true_upper_bounds

    def _ensure_true_bounds(self) -> None:
        """Ensure true bounds (allowing ±inf) are computed and tightened per strategy."""
        if self._true_lower_bounds is not None and self._true_upper_bounds is not None:
            if not self._auto_tightening_done:
                self._maybe_auto_tighten()
            return

        n_var = self.var_manager.n_variables
        sorted_symbols = self.var_manager.sorted_symbols
        symbol_to_index = self.var_manager.symbol_to_index

        # Initialize bounds with infinities
        xl = np.full(n_var, -np.inf)
        xu = np.full(n_var, np.inf)

        # Handle discrete variable constraints (FiniteSet)
        discrete_vars = {}
        if self.con_analyzer:
            for con_item in self.con_analyzer.discrete_constraints:
                if not con_item.free_symbols:
                    continue
                symbol = con_item.free_symbols[0]
                if symbol not in symbol_to_index:
                    continue

                domain = con_item.discrete_domain
                if not domain:
                    continue

                var_idx = symbol_to_index[symbol]

                # 转换为数值类型
                numeric_values = []
                for v in domain.values:
                    try:
                        if float(v) == float(int(v)):
                            numeric_values.append(int(v))
                        else:
                            numeric_values.append(float(v))
                    except Exception:
                        # If cannot convert, keep original but skip bound update
                        continue

                if not numeric_values:
                    continue

                discrete_vars[var_idx] = numeric_values
                xl[var_idx] = min(numeric_values)
                xu[var_idx] = max(numeric_values)

        # Extract simple bounds from constraints (e.g., x >= a, x <= b, Contains(x, Interval(a,b)))
        try:
            if self.con_analyzer:
                from sympy import Eq
                from sympy.core.relational import GreaterThan, StrictGreaterThan, LessThan, StrictLessThan
                ge_types = (GreaterThan, StrictGreaterThan)
                le_types = (LessThan, StrictLessThan)
                from sympy.sets.contains import Contains
                from sympy import Interval

                for con in self.con_analyzer.constraints:
                    # Interval: Contains(x, Interval(a,b))
                    if isinstance(con, Contains):
                        elem, set_obj = con.args
                        if isinstance(set_obj, Interval) and elem in symbol_to_index:
                            i = symbol_to_index[elem]
                            a = float(set_obj.start)
                            b = float(set_obj.end)
                            # Interval 左右端点或许开闭，保守处理为包含
                            xl[i] = max(xl[i], a) if np.isfinite(xl[i]) else a
                            xu[i] = min(xu[i], b) if np.isfinite(xu[i]) else b
                        continue

                    # Relational ops: try extracting simple bounds of Symbol vs numeric
                    if isinstance(con, ge_types + le_types + (Eq,)):
                        lhs, rhs = con.lhs, con.rhs

                        def _convert_to_float(value):
                            if isinstance(value, (int, float, np.integer, np.floating)):
                                return float(value)
                            try:
                                evaluated = value.evalf() if hasattr(value, 'evalf') else value
                                if hasattr(evaluated, 'is_real') and evaluated.is_real is False:
                                    return None
                                return float(evaluated)
                            except Exception:
                                try:
                                    return float(value)
                                except Exception:
                                    return None

                        def _is_number(x):
                            return _convert_to_float(x) is not None

                        # 标准化成 (sym, value, kind)
                        simple = None
                        con_type = type(con)

                        if lhs in symbol_to_index and _is_number(rhs):
                            simple = (lhs, _convert_to_float(rhs), con_type)
                        elif rhs in symbol_to_index and _is_number(lhs):
                            # 反向形式，如 a <= x
                            # 转换为 x >= a 等价更新
                            # 记录成 (sym, value, swapped_type)
                            swapped = {
                                GreaterThan: LessThan,
                                StrictGreaterThan: LessThan,
                                LessThan: GreaterThan,
                                StrictLessThan: GreaterThan,
                                Eq: Eq,
                            }
                            simple = (rhs, _convert_to_float(lhs), swapped.get(con_type, con_type))

                        if simple is None:
                            continue

                        sym, val, ctype = simple
                        if val is None:
                            continue
                        idx = symbol_to_index[sym]

                        if issubclass(ctype, ge_types):
                            # 下界
                            xl[idx] = max(xl[idx], val) if np.isfinite(xl[idx]) else val
                        elif issubclass(ctype, le_types):
                            # 上界
                            xu[idx] = min(xu[idx], val) if np.isfinite(xu[idx]) else val
                        elif ctype is Eq:
                            xl[idx] = xu[idx] = val
        except Exception:
            # 保守：不因异常中断，但增加一次兜底扫描（仅处理最常见的 Ge/Le/Interval 情况）
            try:
                if self.con_analyzer:
                    from sympy import Ge, Gt, Le, Lt, Eq
                    from sympy.sets.contains import Contains
                    from sympy import Interval

                    for con in self.con_analyzer.constraints:
                        # Interval: Contains(x, Interval(a,b))
                        if isinstance(con, Contains):
                            elem, set_obj = con.args
                            if isinstance(set_obj, Interval) and elem in symbol_to_index:
                                i = symbol_to_index[elem]
                                try:
                                    a = float(set_obj.start)
                                    b = float(set_obj.end)
                                    xl[i] = max(xl[i], a) if np.isfinite(xl[i]) else a
                                    xu[i] = min(xu[i], b) if np.isfinite(xu[i]) else b
                                except Exception:
                                    pass
                            continue

                        if isinstance(con, (Ge, Gt, Le, Lt, Eq)):
                            lhs, rhs = con.lhs, con.rhs
                            # 标准化成 (sym, val, sense)
                            triples = []
                            for sym_side, val_side in ((lhs, rhs), (rhs, lhs)):
                                try:
                                    if sym_side in symbol_to_index:
                                        val = float(val_side)
                                        triples.append((sym_side, val, type(con)))
                                        break
                                except Exception:
                                    continue
                            for sym, val, ctype in triples:
                                idx = symbol_to_index[sym]
                                if ctype.__name__ in ('Ge', 'Gt'):
                                    xl[idx] = max(xl[idx], val) if np.isfinite(xl[idx]) else val
                                elif ctype.__name__ in ('Le', 'Lt'):
                                    xu[idx] = min(xu[idx], val) if np.isfinite(xu[idx]) else val
                                elif ctype.__name__ == 'Eq':
                                    xl[idx] = xu[idx] = val
            except Exception:
                # 彻底放弃兜底扫描
                pass

        self._true_lower_bounds = xl
        self._true_upper_bounds = xu
        self._discrete_variables = discrete_vars
        self._auto_tightening_done = False
        self._maybe_auto_tighten()

    def _maybe_auto_tighten(self) -> None:
        """根据配置在首次请求时自动执行紧化"""
        if self._auto_tightening_done:
            return
        self._apply_tightening(self._tighten_strategy, auto_run=True)

    def tighten_bounds(self, strategy: Optional[str] = None) -> bool:
        """
        主动触发边界紧化

        Args:
            strategy: 'rbc'、'lp' 或 'none'，默认沿用初始化配置
        Returns:
            bool: 是否有变量边界被更新
        """
        self._ensure_true_bounds()
        chosen = strategy or self._tighten_strategy
        if not isinstance(chosen, str):
            chosen = self._tighten_strategy
        chosen = chosen.lower()
        if chosen == 'auto':
            chosen = 'rbc'
        return self._apply_tightening(chosen, auto_run=False)

    def update_bounds_from_constraints(self, simple_bound_constraints):
        """
        从简单边界约束更新边界

        Args:
            simple_bound_constraints: 简单边界约束列表
        """
        if self._true_lower_bounds is None or self._true_upper_bounds is None:
            self.extract_bounds()

        # 这个方法将由ConstraintStandardizer调用
        pass

    def _apply_tightening(self, strategy: str, auto_run: bool) -> bool:
        """执行具体的紧化策略"""
        normalized = (strategy or 'none').lower()
        if normalized == 'auto':
            normalized = 'rbc'
        if normalized not in {'none', 'rbc', 'lp'}:
            normalized = 'rbc'

        if normalized == 'none':
            if auto_run:
                self._auto_tightening_done = True
            return False

        linear_rows = self._get_linear_rows()
        if not linear_rows:
            if auto_run:
                self._auto_tightening_done = True
            return False

        changed = False
        if normalized in {'rbc', 'lp'}:
            changed |= self._apply_rbc(linear_rows)
        if normalized == 'lp':
            changed |= self._apply_lp(linear_rows)

        if changed:
            self._invalidate_search_bounds()

        if auto_run:
            self._auto_tightening_done = True
        return changed

    def _get_linear_rows(self) -> List[Tuple[np.ndarray, float]]:
        """收集可用于紧化的线性不等式行，形式为 a^T x <= b"""
        if self._linear_rows_cache is not None:
            return self._linear_rows_cache
        rows: List[Tuple[np.ndarray, float]] = []
        if not self.con_analyzer:
            self._linear_rows_cache = rows
            return rows
        processed: List[Any] = []
        try:
            processed.extend(self.con_analyzer.inequality_constraints or [])
        except Exception:
            pass
        try:
            equality_constraints = self.con_analyzer.equality_constraints or []
        except Exception:
            equality_constraints = []
        for con in processed:
            rows.extend(self._linear_rows_from_constraint(con))
        for con in equality_constraints:
            rows.extend(self._linear_rows_from_constraint(con, treat_as_eq=True))
        self._linear_rows_cache = rows
        return rows

    def _linear_rows_from_constraint(self, ir_constraint, treat_as_eq: bool = False) -> List[Tuple[np.ndarray, float]]:
        """将IR约束转换为一个或多个 a^T x <= b 不等式"""
        if ir_constraint is None:
            return []
        expr = getattr(ir_constraint, 'normalized_expr', None)
        if expr is None and getattr(ir_constraint, 'lhs', None) is not None and getattr(ir_constraint, 'rhs', None) is not None:
            try:
                expr = (ir_constraint.lhs - ir_constraint.rhs).expand()
            except Exception:
                expr = ir_constraint.lhs - ir_constraint.rhs
        if expr is None:
            return []
        linear_data = self._extract_linear_data(expr)
        if linear_data is None:
            return []
        coeffs, const_term = linear_data
        if not coeffs.size or np.all(np.abs(coeffs) <= self._coeff_tol):
            return []

        rhs_le = -const_term
        rows: List[Tuple[np.ndarray, float]] = []
        sense = getattr(ir_constraint, 'sense', None)
        sense_value = getattr(sense, 'value', sense) if sense is not None else None
        if treat_as_eq or sense_value == IRConstraintSense.EQ:
            rows.append((coeffs.copy(), float(rhs_le)))
            rows.append((-coeffs.copy(), float(const_term)))
            return rows
        if sense_value == IRConstraintSense.LE:
            rows.append((coeffs.copy(), float(rhs_le)))
        elif sense_value == IRConstraintSense.GE:
            rows.append((-coeffs.copy(), float(const_term)))
        return rows

    def _extract_linear_data(self, expr) -> Optional[Tuple[np.ndarray, float]]:
        """解析线性表达式，返回系数向量和常数项"""
        try:
            expanded = sp.expand(expr)
        except Exception:
            expanded = expr
        symbols = self.var_manager.sorted_symbols
        if not symbols:
            return None
        try:
            poly = sp.Poly(expanded, *symbols)
        except Exception:
            return None
        if poly.total_degree() > 1:
            return None

        coeffs = np.zeros(len(symbols), dtype=float)
        const_term = 0.0
        try:
            terms = poly.terms()
        except Exception:
            return None
        for monom, coeff in terms:
            power_sum = sum(monom)
            coeff_val = float(coeff)
            if power_sum == 0:
                const_term = coeff_val
                continue
            if power_sum != 1:
                return None
            target_idx = None
            for idx, power in enumerate(monom):
                if power == 1:
                    target_idx = idx
                    break
                if power not in (0, 1):
                    target_idx = None
                    break
            if target_idx is None:
                return None
            coeffs[target_idx] += coeff_val
        return coeffs, const_term

    def _apply_rbc(self, linear_rows: List[Tuple[np.ndarray, float]]) -> bool:
        """使用区间一致性方法迭代收紧线性不等式的变量界"""
        if self._true_lower_bounds is None or self._true_upper_bounds is None:
            return False
        xl = self._true_lower_bounds
        xu = self._true_upper_bounds
        n_var = len(xl)
        if n_var == 0 or not linear_rows:
            return False

        max_iter = int(self._tighten_options.get('rbc_max_iter', 25))
        tol = float(self._tighten_options.get('rbc_tol', 1e-9))
        max_iter = max(1, max_iter)
        changed = False

        for _ in range(max_iter):
            iter_changed = False
            for coeffs, rhs in linear_rows:
                if coeffs.size != n_var:
                    continue
                rhs_val = float(rhs)
                for idx in range(n_var):
                    a_i = float(coeffs[idx])
                    if abs(a_i) <= self._coeff_tol:
                        continue
                    if a_i > 0:
                        min_sum = self._accumulate_linear_sum(coeffs, idx, xl, xu, use_min=True)
                        if min_sum is None:
                            continue
                        candidate = (rhs_val - min_sum) / a_i
                        if not np.isfinite(candidate):
                            continue
                        if np.isfinite(xu[idx]):
                            if candidate + tol < xu[idx]:
                                xu[idx] = candidate
                                iter_changed = True
                        else:
                            xu[idx] = candidate
                            iter_changed = True
                    else:
                        max_sum = self._accumulate_linear_sum(coeffs, idx, xl, xu, use_min=False)
                        if max_sum is None:
                            continue
                        candidate = (rhs_val - max_sum) / a_i
                        if not np.isfinite(candidate):
                            continue
                        if np.isfinite(xl[idx]):
                            if candidate - tol > xl[idx]:
                                xl[idx] = candidate
                                iter_changed = True
                        else:
                            xl[idx] = candidate
                            iter_changed = True
            if not iter_changed:
                break
            changed = True

        if changed:
            self._enforce_bound_consistency()
        return changed

    def _accumulate_linear_sum(
        self,
        coeffs: np.ndarray,
        excluded_index: int,
        xl: np.ndarray,
        xu: np.ndarray,
        use_min: bool
    ) -> Optional[float]:
        """根据当前区间计算除某变量外的线性组合上下界"""
        total = 0.0
        for j, coeff in enumerate(coeffs):
            if j == excluded_index or abs(coeff) <= self._coeff_tol:
                continue
            lb = xl[j]
            ub = xu[j]
            contrib = (
                self._compute_min_contribution(coeff, lb, ub)
                if use_min else
                self._compute_max_contribution(coeff, lb, ub)
            )
            if contrib is None:
                return None
            total += contrib
        return total

    def _compute_min_contribution(self, coeff: float, lb: float, ub: float) -> Optional[float]:
        """给定系数和变量区间，求该项的最小贡献"""
        target = lb if coeff >= 0 else ub
        if target is None or not np.isfinite(target):
            return None
        return coeff * target

    def _compute_max_contribution(self, coeff: float, lb: float, ub: float) -> Optional[float]:
        """给定系数和变量区间，求该项的最大贡献"""
        target = ub if coeff >= 0 else lb
        if target is None or not np.isfinite(target):
            return None
        return coeff * target

    def _enforce_bound_consistency(self) -> None:
        """若紧化后出现 lb > ub，进行保守修正"""
        if self._true_lower_bounds is None or self._true_upper_bounds is None:
            return
        tol = float(self._tighten_options.get('bound_consistency_tol', 1e-8))
        symbols = self.var_manager.sorted_symbols
        for idx in range(len(self._true_lower_bounds)):
            lb = self._true_lower_bounds[idx]
            ub = self._true_upper_bounds[idx]
            if np.isfinite(lb) and np.isfinite(ub):
                if lb <= ub + tol:
                    continue
                midpoint = 0.5 * (lb + ub)
                warnings.warn(
                    f"变量'{symbols[idx]}' 的紧化结果出现 lb > ub，已回退为中点 {midpoint:.6g}。",
                    RuntimeWarning
                )
                self._true_lower_bounds[idx] = midpoint
                self._true_upper_bounds[idx] = midpoint
            elif lb > ub:
                # 其中一端为无穷，直接将另一端复制过去
                warnings.warn(
                    f"变量'{symbols[idx]}' 的紧化结果出现无序边界，已做保守修正。",
                    RuntimeWarning
                )
                if not np.isfinite(lb):
                    self._true_lower_bounds[idx] = ub
                else:
                    self._true_upper_bounds[idx] = lb

    def _apply_lp(self, linear_rows: List[Tuple[np.ndarray, float]]) -> bool:
        """借助轻量LP求解器做进一步紧化"""
        if self._true_lower_bounds is None or self._true_upper_bounds is None:
            return False
        n_var = len(self._true_lower_bounds)
        if n_var == 0 or not linear_rows:
            return False

        max_vars = int(self._tighten_options.get('lp_max_vars', 40))
        max_cons = int(self._tighten_options.get('lp_max_constraints', 200))
        if n_var > max_vars or len(linear_rows) > max_cons:
            return False

        try:
            import pyomo.environ as pyo
            from pyomo.opt import SolverStatus, TerminationCondition
        except Exception:
            warnings.warn("未检测到Pyomo，无法执行 LP 紧化。")
            return False

        solver = self._choose_lp_solver(pyo)
        if solver is None:
            return False

        model = pyo.ConcreteModel()
        model.idx = pyo.Set(initialize=list(range(n_var)))
        model.x = pyo.Var(model.idx)
        for i in range(n_var):
            lb = self._true_lower_bounds[i]
            ub = self._true_upper_bounds[i]
            model.x[i].setlb(None if not np.isfinite(lb) else float(lb))
            model.x[i].setub(None if not np.isfinite(ub) else float(ub))

        model.lin_cons = pyo.ConstraintList()
        for coeffs, rhs in linear_rows:
            if coeffs.size != n_var:
                continue
            expr = sum(float(coeffs[j]) * model.x[j] for j in range(n_var) if abs(coeffs[j]) > self._coeff_tol)
            if expr is None:
                continue
            try:
                model.lin_cons.add(expr <= float(rhs))
            except Exception:
                continue

        tol = float(self._tighten_options.get('lp_tol', 1e-9))
        changed = False

        def _solve_bound(var_index: int, sense: str) -> Optional[float]:
            if hasattr(model, '_tight_obj'):
                model.del_component(model._tight_obj)
            if sense == 'min':
                model._tight_obj = pyo.Objective(expr=model.x[var_index], sense=pyo.minimize)
            else:
                model._tight_obj = pyo.Objective(expr=model.x[var_index], sense=pyo.maximize)
            try:
                result = solver.solve(model, tee=False)
            except Exception:
                model.del_component(model._tight_obj)
                return None
            status = getattr(result.solver, 'status', None)
            term = getattr(result.solver, 'termination_condition', None)
            if status not in {SolverStatus.ok, SolverStatus.warning}:
                model.del_component(model._tight_obj)
                return None
            if term not in {TerminationCondition.optimal, TerminationCondition.feasible}:
                model.del_component(model._tight_obj)
                return None
            value = pyo.value(model.x[var_index])
            model.del_component(model._tight_obj)
            if value is None or not np.isfinite(value):
                return None
            return float(value)

        for i in range(n_var):
            min_val = _solve_bound(i, 'min')
            if min_val is not None:
                current_lb = self._true_lower_bounds[i]
                if not np.isfinite(current_lb) or min_val - current_lb > tol:
                    self._true_lower_bounds[i] = min_val
                    changed = True
            max_val = _solve_bound(i, 'max')
            if max_val is not None:
                current_ub = self._true_upper_bounds[i]
                if not np.isfinite(current_ub) or current_ub - max_val > tol:
                    self._true_upper_bounds[i] = max_val
                    changed = True

        if hasattr(model, '_tight_obj'):
            model.del_component(model._tight_obj)

        if changed:
            self._enforce_bound_consistency()
        return changed

    def _choose_lp_solver(self, pyo_module):
        """按候选列表选择可用的线性规划求解器"""
        solver_hint = self._tighten_options.get('lp_solver')
        candidates = []
        if solver_hint:
            candidates.append(str(solver_hint))
        fallback = self._tighten_options.get('lp_solver_candidates')
        if isinstance(fallback, (list, tuple)):
            candidates.extend(str(item) for item in fallback)
        else:
            candidates.extend(['glpk', 'cbc'])

        seen = set()
        for name in candidates:
            if not name or name in seen:
                continue
            seen.add(name)
            try:
                solver = pyo_module.SolverFactory(name)
            except Exception:
                continue
            if solver is None or not solver.available(False):
                continue
            time_limit = self._tighten_options.get('lp_time_limit')
            if time_limit is not None:
                try:
                    solver.options['timelimit'] = float(time_limit)
                except Exception:
                    pass
            return solver

        warnings.warn("LP 紧化所需的求解器不可用，已跳过。")
        return None

    def check_all_variables_bounded(self) -> bool:
        """
        检查是否所有变量都同时具有上下界约束

        Returns:
            bool: 是否所有变量都有上下界
        """
        # 基于已提取的上下界直接判断是否无限
        xl, xu = self.extract_bounds()
        return (np.isfinite(xl).all() and np.isfinite(xu).all())

    @property
    def lower_bounds(self) -> np.ndarray:
        """获取下界数组"""
        self._ensure_true_bounds()
        return self._true_lower_bounds

    @property
    def upper_bounds(self) -> np.ndarray:
        """获取上界数组"""
        self._ensure_true_bounds()
        return self._true_upper_bounds

    @property
    def discrete_variables(self) -> Dict:
        """获取离散变量字典"""
        if self._discrete_variables is None:
            self.extract_bounds()
        return self._discrete_variables

    @property
    def search_lower_bounds(self) -> np.ndarray:
        """兼容占位：返回真实下界。"""
        self._ensure_true_bounds()
        return self._true_lower_bounds

    @property
    def search_upper_bounds(self) -> np.ndarray:
        """返回真实上界。"""
        self._ensure_true_bounds()
        return self._true_upper_bounds

    @property
    def search_bound_annotations(self) -> tuple:
        """搜索盒注释（当前版本返回空元组）。"""
        return tuple()
        # 外部边界映射：来自功能型配置（控制系数/初值/常数等）
        self._external_bounds = dict(external_bounds or {})
        # 先应用外部边界（若提供）
        if self._external_bounds:
            for sym, (lb_val, ub_val) in self._external_bounds.items():
                try:
                    idx = symbol_to_index.get(sym)
                    if idx is None:
                        # 尝试按名称匹配
                        for s, i in symbol_to_index.items():
                            if str(s) == str(sym):
                                idx = i
                                break
                    if idx is None:
                        continue
                    if lb_val is not None:
                        xl[idx] = float(lb_val)
                    if ub_val is not None:
                        xu[idx] = float(ub_val)
                except Exception:
                    continue
