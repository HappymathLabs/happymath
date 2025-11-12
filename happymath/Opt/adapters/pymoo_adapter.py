"""
Pymoo适配器

基于统一IR结构构建Pymoo问题对象，支持连续/整数/枚举变量、
标准代数约束以及函数式目标的求值。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import sympy as sp
from pymoo.core.problem import Problem

from .constraint_handlers.epsilon_estimator import EpsilonEstimator
from ..ir import IRConstraintCategory, IRConstraintSense, IROptVarType
from ..functional.evaluator import TrajectoryEvaluator, _build_metric_specs_from_meta
from ..functional.config import MetricSpec, IntegrandSpec


class PymooAdapter:
    """Pymoo问题适配器"""

    def __init__(self, parse_result, epsilon: float = 1e-6,
                 # 修复与可行性相关配置（方案1、2、4）
                 repair_max_steps: int = 3,
                 repair_trust_radius_factor: float = 0.1,
                 repair_linesearch: bool = True,
                 repair_linesearch_max_steps: int = 5,
                 repair_linesearch_c: float = 1e-4,
                 enable_normalization: bool = False,
                 strict_ieq_enable: bool = True,
                 strict_ieq_topk: int = 2,
                 strict_ieq_step_factor: float = 0.2):
        """
        初始化Pymoo适配器

        Args:
            parse_result: ParseResult实例
            epsilon: 严格不等式的默认epsilon
        """
        self.parse_result = parse_result
        self.ir_problem = parse_result.ir_problem
        self.epsilon = float(epsilon)
        self.epsilon_estimator = EpsilonEstimator()
        self.symbol_to_index = parse_result.symbol_to_index
        self.sorted_symbols = parse_result.sorted_symbols
        # 修复参数
        self.repair_max_steps = int(repair_max_steps or 0)
        self.repair_trust_radius_factor = float(repair_trust_radius_factor or 0.0)
        self.repair_linesearch = bool(repair_linesearch)
        self.repair_linesearch_max_steps = int(repair_linesearch_max_steps or 0)
        self.repair_linesearch_c = float(repair_linesearch_c or 1e-4)
        self.enable_normalization = bool(enable_normalization)
        self.strict_ieq_enable = bool(strict_ieq_enable)
        self.strict_ieq_topk = int(strict_ieq_topk or 1)
        self.strict_ieq_step_factor = float(strict_ieq_step_factor or 0.1)

    def convert(self) -> Problem:
        """构建并返回Pymoo问题对象"""
        data = self._prepare_problem_data()
        adapter = self

        class IROptPymooProblem(Problem):
            def __init__(self):
                super().__init__(
                    n_var=data['n_var'],
                    n_obj=data['n_obj'],
                    n_ieq_constr=data['n_ieq'],
                    n_eq_constr=data['n_eq'],
                    xl=data['xl'],
                    xu=data['xu'],
                    vtype=data['vtype']
                )
                # 暴露用于结果解码的元数据
                self._enum_maps = data['enum_maps']
                self._var_types = data['var_types']
                self._symbol_order = data['symbol_order']
                self._symbol_to_index = data['symbol_to_index']
                self._search_box_annotations = data.get('search_bound_annotations') or tuple()
                self._search_bound_range = data.get('search_bound_range')
                # 评估预算提示（供上层优化器使用，可被覆盖）
                n_constr_total = int(data['n_ieq'] + data['n_eq'])
                if self.n_obj == 1:
                    hint = 200 + 50 * int(self.n_var) + 20 * n_constr_total
                    self._budget_hint = int(min(1000, max(200, hint)))
                else:
                    hint = 400 + 80 * int(self.n_var) + 40 * n_constr_total
                    self._budget_hint = int(min(3000, max(400, hint)))
                # 暴露软等式阈值状态，便于上层联动成功判据
                self._soft_eq_state = data.get('soft_eq_state')
                # 功能型配置：准备一次仿真评估器
                self._functional_cfg = getattr(adapter.parse_result, '_functional_config', None)
                self._traj_evaluator = TrajectoryEvaluator(self._functional_cfg) if self._functional_cfg else None

            def _evaluate(problem_self, x, out, *args, **kwargs):
                adapter._evaluate_problem(x, out, data, getattr(problem_self, '_traj_evaluator', None), getattr(problem_self, '_functional_cfg', None))

            def _repair(problem_self, X, **kwargs):
                return adapter._repair_decision_vectors(X, data)

        return IROptPymooProblem()

    # === 数据准备 ===

    def _prepare_problem_data(self) -> Dict[str, Any]:
        """准备Pymoo构建所需的元数据"""
        variables = self.ir_problem.variables
        objectives = self.ir_problem.objectives
        constraints = self.ir_problem.constraints

        # 仅针对 IR 中暴露的决策变量构建边界（优先使用 IR 变量自身的边界定义）
        n_var = len(variables)
        xl = np.full(n_var, -np.inf, dtype=float)
        xu = np.full(n_var,  np.inf, dtype=float)
        for i, ir_var in enumerate(variables):
            try:
                if ir_var.lower_bound is not None:
                    xl[i] = float(ir_var.lower_bound)
                if ir_var.upper_bound is not None:
                    xu[i] = float(ir_var.upper_bound)
            except Exception:
                pass
        # 不再注入默认搜索盒；诊断保持为空（严格模式由后续检查保障）
        search_annotations = tuple()
        search_range = None

        var_types: List[IROptVarType] = []
        enum_maps: Dict[int, Tuple[float, ...]] = {}
        dtype_tokens: List[str] = []

        for idx, ir_var in enumerate(variables):
            var_types.append(ir_var.var_type)
            if ir_var.var_type == IROptVarType.CONTINUOUS:
                dtype_tokens.append('float')
            elif ir_var.var_type == IROptVarType.BINARY:
                xl[idx] = 0.0
                xu[idx] = 1.0
                dtype_tokens.append('int')
            elif ir_var.var_type == IROptVarType.INTEGER:
                if ir_var.lower_bound is not None:
                    xl[idx] = float(ir_var.lower_bound)
                if ir_var.upper_bound is not None:
                    xu[idx] = float(ir_var.upper_bound)
                dtype_tokens.append('int')
            elif ir_var.var_type == IROptVarType.ENUM:
                domain = ir_var.discrete_domain.values if ir_var.discrete_domain else ()
                numeric_values = []
                seen = set()
                for v in domain:
                    try:
                        num = float(v)
                    except Exception as exc:
                        raise ValueError(f"离散变量 {ir_var.name} 的值 {v} 无法转换为浮点数") from exc
                    if num not in seen:
                        numeric_values.append(num)
                        seen.add(num)
                if not numeric_values:
                    raise ValueError(f"枚举变量 {ir_var.name} 缺少可用取值")
                enum_maps[idx] = tuple(numeric_values)
                xl[idx] = 0.0
                xu[idx] = float(len(numeric_values) - 1)
                dtype_tokens.append('int')
            else:
                dtype_tokens.append('float')

        # 严格模式：所有变量需具有有限上下界（在类型规范化之后再检查）
        if not (np.isfinite(xl).all() and np.isfinite(xu).all()):
            unbounded = []
            for idx, ir_var in enumerate(variables):
                if not (np.isfinite(xl[idx]) and np.isfinite(xu[idx])):
                    unbounded.append(ir_var.name)
            details = ", ".join(unbounded) if unbounded else "(未知变量)"
            raise ValueError(
                "Pymoo 严格模式：检测到变量缺少有限上下界，无法构建启发式问题。"
                f" 未界定变量：{details}"
            )

        if all(token == 'float' for token in dtype_tokens):
            vtype = float
        elif all(token == 'int' for token in dtype_tokens):
            vtype = int
        else:
            vtype = np.array([int if token == 'int' else float for token in dtype_tokens])

        inequality_constraints = [
            con for con in constraints
            if (con.category == IRConstraintCategory.ALGEBRAIC and con.sense in {IRConstraintSense.LE, IRConstraintSense.GE})
        ]
        # 将 LOGICAL（Piecewise/Indicator）约束视作不等式约束，评估时返回 g(x)≤0
        logical_constraints = [
            con for con in constraints
            if con.category == IRConstraintCategory.LOGICAL
        ]
        inequality_constraints.extend(logical_constraints)

        equality_constraints = [
            con for con in constraints
            if (con.category == IRConstraintCategory.ALGEBRAIC and con.sense == IRConstraintSense.EQ)
        ]
        # 将 FUNCTIONAL 约束纳入评估集合（按 sense 分流）
        functional_constraints = [con for con in constraints if con.category == IRConstraintCategory.FUNCTIONAL]
        for con in functional_constraints:
            if con.sense == IRConstraintSense.EQ:
                equality_constraints.append(con)
            elif con.sense in {IRConstraintSense.LE, IRConstraintSense.GE}:
                inequality_constraints.append(con)

        # 基于 IR 变量列表构建内部符号顺序与索引映射（剔除仿真自变量，如 t）
        symbol_order = [var.symbol for var in variables]
        reduced_sym2idx = {sym: i for i, sym in enumerate(symbol_order)}

        n_soft_eq = len(equality_constraints)
        soft_eq_state = self._init_soft_eq_state(len(variables))
        linear_projection = self._build_linear_equality_projection(equality_constraints, symbol_order)
        nonlinear_eq_specs = self._build_nonlinear_eq_specs(equality_constraints, reduced_sym2idx)

        # === 功能型指标缓存：统一收集“目标 + 功能型约束”的评估指标 ===
        func_metrics: List[Any] = []
        func_obj_metric_ids: List[str] = []
        try:
            functional_cfg = getattr(self.parse_result, '_functional_config', None)
            if functional_cfg is not None:
                # 1) 基础：cfg.metrics（若提供）
                func_metrics = list(_build_metric_specs_from_meta(functional_cfg))
                # 2) 从功能型约束元信息生成对应指标，id 规范：con:{identifier}
                cmeta = getattr(functional_cfg, 'constraint_meta', {}) if not isinstance(functional_cfg, dict) else (functional_cfg.get('constraint_meta') or {})
                if isinstance(cmeta, dict):
                    for key, meta in cmeta.items():
                        agg = str(meta.get('aggregation', '')).lower()
                        if agg == 'integral':
                            expr = meta.get('expr')
                            func_metrics.append(MetricSpec(id=f"con:{key}", kind='integral', integrand=IntegrandSpec(id=f"con:{key}:integrand", expr=expr)))
                        elif agg in {'final_state', 'terminal'}:
                            func_metrics.append(MetricSpec(id=f"con:{key}", kind='terminal', state_index=int(meta.get('state_index', 0))))
                        else:
                            func_metrics.append(MetricSpec(id=f"con:{key}", kind='path', state_index=int(meta.get('state_index', 0)), agg=agg or 'l2_norm'))
                # 3) 目标→指标 id 映射：优先 obj:{index}，否则退回第一个 obj:* 指标
                obj_ids = [m.id for m in func_metrics if hasattr(m, 'id') and isinstance(getattr(m, 'id'), str) and str(getattr(m, 'id')).startswith('obj:')]
                for j, obj in enumerate(objectives):
                    idx = int(obj.metadata.get('index', j)) if isinstance(obj.metadata, dict) else j
                    candidate = f"obj:{idx}"
                    if candidate in obj_ids:
                        func_obj_metric_ids.append(candidate)
                    else:
                        func_obj_metric_ids.append(obj_ids[0] if obj_ids else '')
        except Exception:
            func_metrics = []
            func_obj_metric_ids = []

        return {
            'n_var': len(variables),
            'n_obj': len(objectives),
            'n_ieq': len(inequality_constraints) + n_soft_eq,
            'n_eq': len(equality_constraints),
            'xl': xl,
            'xu': xu,
            'vtype': vtype,
            'variables': variables,
            'objectives': objectives,
            'inequality_constraints': inequality_constraints,
            'equality_constraints': equality_constraints,
            'n_soft_eq': n_soft_eq,
            'var_types': var_types,
            'enum_maps': enum_maps,
            'symbol_to_index': reduced_sym2idx,
            'symbol_order': symbol_order,
            'logical_constraints': logical_constraints,
            'search_bound_annotations': search_annotations,
            'search_bound_range': search_range,
            'soft_eq_state': soft_eq_state,
            'linear_eq_projection': linear_projection,
            'nonlinear_eq_specs': nonlinear_eq_specs,
            # 功能型缓存相关
            'functional_metrics': tuple(func_metrics) if func_metrics else tuple(),
            'func_obj_metric_ids': tuple(func_obj_metric_ids) if func_obj_metric_ids else tuple(),
        }

    # === 求值逻辑 ===

    def _evaluate_problem(self, x, out: Dict[str, Any], data: Dict[str, Any], traj_eval: TrajectoryEvaluator | None, functional_cfg: Any | None) -> None:
        """对候选解进行评估"""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        decision_values = self._map_decision_variables(x, data)
        # 若存在 FUNCTIONAL 目标/约束，先合并一次仿真评估所有指标
        func_cache: List[Dict[str, float]] | None = None
        if traj_eval is not None and functional_cfg is not None:
            func_cache = []
            # 构造统一指标集（覆盖目标与功能型约束）
            metrics = data.get('functional_metrics') or _build_metric_specs_from_meta(functional_cfg)
            for row in range(decision_values.shape[0]):
                opt_vars = self._build_opt_vars(decision_values[row], data)
                func_cache.append(traj_eval.evaluate_all(opt_vars, metrics))

        f = self._evaluate_objectives(decision_values, data, func_cache)
        out["F"] = f

        soft_state = data.get('soft_eq_state')
        soft_tau = None
        if soft_state:
            soft_tau = self._update_soft_eq_state(soft_state, decision_values.shape[0])
        g, h = self._evaluate_constraints(decision_values, data, soft_tau, func_cache)
        if g is not None:
            out["G"] = g
        if h is not None:
            out["H"] = h

    def _map_decision_variables(self, x: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
        """将原始决策向量映射到实际变量取值"""
        values = x.astype(float, copy=True)
        var_types = data['var_types']
        enum_maps = data['enum_maps']

        for idx, var_type in enumerate(var_types):
            if var_type == IROptVarType.ENUM:
                mapping = enum_maps[idx]
                upper = len(mapping) - 1
                for row in range(values.shape[0]):
                    choice = int(np.clip(np.round(values[row, idx]), 0, upper))
                    values[row, idx] = mapping[choice]
            elif var_type == IROptVarType.BINARY:
                values[:, idx] = np.clip(np.round(values[:, idx]), 0, 1)
            elif var_type == IROptVarType.INTEGER:
                values[:, idx] = np.round(values[:, idx])

        return values

    def _evaluate_objectives(self, values: np.ndarray, data: Dict[str, Any], func_cache: List[Dict[str, float]] | None = None) -> np.ndarray:
        """评估目标函数"""
        objectives = data['objectives']
        symbol_to_index = data['symbol_to_index']
        batch_size = values.shape[0]
        f = np.zeros((batch_size, len(objectives)))

        for obj_idx, obj in enumerate(objectives):
            for row in range(batch_size):
                if obj.is_functional and obj.functional_spec is not None:
                    if func_cache is not None:
                        # 从缓存按映射 id 读取；若缺失则回退到 evaluator
                        id_map = data.get('func_obj_metric_ids') or ()
                        metric_id = None
                        try:
                            metric_id = id_map[obj_idx] if obj_idx < len(id_map) else None
                        except Exception:
                            metric_id = None
                        if metric_id:
                            val = float(func_cache[row].get(metric_id, 0.0))
                        else:
                            opt_vars = self._build_opt_vars(values[row], data)
                            val = self._evaluate_functional_objective(obj, opt_vars)
                    else:
                        opt_vars = self._build_opt_vars(values[row], data)
                        val = self._evaluate_functional_objective(obj, opt_vars)
                else:
                    lambda_func = obj.lambda_func
                    if lambda_func is not None:
                        args = [values[row, symbol_to_index[sym]] for sym in obj.free_symbols]
                        val = float(lambda_func(*args))
                    else:
                        substitutions = {sym: values[row, symbol_to_index[sym]] for sym in obj.free_symbols}
                        val = float(obj.expression.subs(substitutions).evalf())

                f[row, obj_idx] = val if obj.sense == 'min' else -val

        return f

    def _evaluate_functional_objective(self, obj, opt_vars: Dict[str, float]) -> float:
        """评估函数式目标"""
        spec = obj.functional_spec
        evaluator = spec.evaluator if spec else None
        if evaluator is None:
            raise RuntimeError("函数式目标缺少可用的评估器（eval_spec.evaluator 未提供）")
        return float(evaluator(opt_vars))

    def _evaluate_constraints(self, values: np.ndarray, data: Dict[str, Any], soft_eq_tau: float | None = None, func_cache: List[Dict[str, float]] | None = None) -> Tuple[np.ndarray | None, np.ndarray | None]:
        """评估约束条件"""
        ineq_constraints = data['inequality_constraints']
        eq_constraints = data['equality_constraints']
        symbol_to_index = data['symbol_to_index']
        batch_size = values.shape[0]
        n_soft_eq = data.get('n_soft_eq', 0)
        total_ineq = len(ineq_constraints) + n_soft_eq
        g = np.zeros((batch_size, total_ineq)) if total_ineq else None
        soft_offset = len(ineq_constraints)
        h = np.zeros((batch_size, len(eq_constraints))) if eq_constraints else None

        for c_idx, ir_con in enumerate(ineq_constraints):
            epsilon = self._resolve_constraint_epsilon(ir_con)
            for row in range(batch_size):
                if ir_con.category == IRConstraintCategory.LOGICAL:
                    opt_vars = self._build_opt_vars(values[row], data)
                    cons_val = self._evaluate_logical_constraint(ir_con, opt_vars, epsilon)
                elif ir_con.category == IRConstraintCategory.FUNCTIONAL:
                    if func_cache is not None:
                        metric_id = f"con:{ir_con.identifier}"
                        if metric_id in func_cache[row]:
                            cons_val = float(func_cache[row].get(metric_id, 0.0))
                        else:
                            opt_vars = self._build_opt_vars(values[row], data)
                            cons_val = float(self._evaluate_functional_constraint(ir_con, opt_vars))
                    else:
                        opt_vars = self._build_opt_vars(values[row], data)
                        cons_val = float(self._evaluate_functional_constraint(ir_con, opt_vars))
                else:
                    if ir_con.lambda_func is not None:
                        args = [values[row, symbol_to_index[sym]] for sym in ir_con.free_symbols]
                        expr_val = float(ir_con.lambda_func(*args))
                    else:
                        expr = ir_con.normalized_expr
                        substitutions = {sym: values[row, symbol_to_index[sym]] for sym in ir_con.free_symbols}
                        expr_val = float(expr.subs(substitutions).evalf())
                    if ir_con.sense == IRConstraintSense.LE:
                        cons_val = expr_val + (epsilon if ir_con.strict else 0.0)
                    else:  # GE
                        cons_val = -expr_val + (epsilon if ir_con.strict else 0.0)
                g[row, c_idx] = cons_val

        for c_idx, ir_con in enumerate(eq_constraints):
            for row in range(batch_size):
                if ir_con.category == IRConstraintCategory.FUNCTIONAL:
                    if func_cache is not None:
                        metric_id = f"con:{ir_con.identifier}"
                        if metric_id in func_cache[row]:
                            expr_val = float(func_cache[row].get(metric_id, 0.0))
                        else:
                            opt_vars = self._build_opt_vars(values[row], data)
                            expr_val = float(self._evaluate_functional_constraint(ir_con, opt_vars))
                    else:
                        opt_vars = self._build_opt_vars(values[row], data)
                        expr_val = float(self._evaluate_functional_constraint(ir_con, opt_vars))
                else:
                    if ir_con.lambda_func is not None:
                        args = [values[row, symbol_to_index[sym]] for sym in ir_con.free_symbols]
                        expr_val = float(ir_con.lambda_func(*args))
                    else:
                        expr = ir_con.normalized_expr
                        substitutions = {sym: values[row, symbol_to_index[sym]] for sym in ir_con.free_symbols}
                        expr_val = float(expr.subs(substitutions).evalf())
                h[row, c_idx] = expr_val
                if g is not None and n_soft_eq:
                    tol = soft_eq_tau if soft_eq_tau is not None else self.epsilon
                    g[row, soft_offset + c_idx] = max(abs(expr_val) - tol, 0.0)

        return g, h

    def _evaluate_functional_constraint(self, ir_con, opt_vars: Dict[str, float]) -> float:
        """计算 FUNCTIONAL 约束并确保返回标量"""
        evaluator = ir_con.functional_spec.evaluator
        result = evaluator(opt_vars)
        agg = getattr(ir_con.functional_spec, "metadata", {}).get("aggregation")
        if agg:
            return self._aggregate_functional_result(result, agg)
        return self._coerce_to_scalar(result, ir_con.identifier)

    def _evaluate_logical_constraint(self, ir_con, opt_vars: Dict[str, float], epsilon: float) -> float:
        """评估 LOGICAL（Piecewise/Indicator）约束，返回 g(x)（≤0 为满足）。

        规则：
        - 对于 Piecewise：找到第一个条件为 True 的分支；
            分支 expr 若为 True → 返回 0；若为 False → 返回 正数（1.0）；
            若为 Relational（Eq/Le/Ge/Gt/Lt）→ 返回相应违反度：
              Le: max(lhs - rhs + eps_strict, 0)
              Ge: max(rhs - lhs + eps_strict, 0)
              Eq: max(abs(lhs - rhs) - tol_eq, 0)
              Gt/Lt 转换为 ≥/≤ 并加严格 epsilon。
        - 若无条件满足且不存在默认 True 分支 → 视作违反（返回 1.0）。
        """
        from sympy import S
        meta = ir_con.metadata or {}
        kind = meta.get('logical_kind')
        if kind == 'piecewise':
            branches = meta.get('branches') or []
            for br in branches:
                cond = br.get('cond')
                if self._eval_bool(cond, self._sympy_subs_map(ir_con, opt_vars)):
                    expr = br.get('expr')
                    if expr is True or expr is S.true:
                        return 0.0
                    if expr is False or expr is S.false:
                        return 1.0
                    return self._relational_violation(expr, self._sympy_subs_map(ir_con, opt_vars), epsilon)
            # 无分支命中，违反
            return 1.0
        # 其他逻辑类型暂不支持，保守返回违反
        return 1.0

    def _relational_violation(self, expr, opt_vars: Dict[str, float], epsilon: float) -> float:
        from sympy import Eq, Ge, Le, Gt, Lt
        if isinstance(expr, Eq):
            lhs = float(expr.lhs.subs(opt_vars).evalf())
            rhs = float(expr.rhs.subs(opt_vars).evalf())
            return max(abs(lhs - rhs) - 1e-6, 0.0)
        elif isinstance(expr, Le):
            lhs = float(expr.lhs.subs(opt_vars).evalf())
            rhs = float(expr.rhs.subs(opt_vars).evalf())
            return max(lhs - rhs + 0.0, 0.0)
        elif isinstance(expr, Ge):
            lhs = float(expr.lhs.subs(opt_vars).evalf())
            rhs = float(expr.rhs.subs(opt_vars).evalf())
            return max(rhs - lhs + 0.0, 0.0)
        elif isinstance(expr, Gt):
            lhs = float(expr.lhs.subs(opt_vars).evalf())
            rhs = float(expr.rhs.subs(opt_vars).evalf())
            return max(rhs - lhs + epsilon, 0.0)
        elif isinstance(expr, Lt):
            lhs = float(expr.lhs.subs(opt_vars).evalf())
            rhs = float(expr.rhs.subs(opt_vars).evalf())
            return max(lhs - rhs + epsilon, 0.0)
        else:
            # 非关系表达式，保守视为违反
            return 1.0

    def _eval_bool(self, cond, opt_vars: Dict[str, float]) -> bool:
        from sympy import Eq, Ge, Le, Gt, Lt
        try:
            if cond is True:
                return True
            if cond is False:
                return False
            # 关系表达式专门处理
            if isinstance(cond, (Eq, Ge, Le, Gt, Lt)):
                lhs = float(cond.lhs.subs(opt_vars).evalf())
                rhs = float(cond.rhs.subs(opt_vars).evalf())
                if isinstance(cond, Eq):
                    return abs(lhs - rhs) <= 1e-8
                if isinstance(cond, Ge):
                    return lhs >= rhs - 1e-12
                if isinstance(cond, Le):
                    return lhs <= rhs + 1e-12
                if isinstance(cond, Gt):
                    return lhs > rhs + 1e-12
                if isinstance(cond, Lt):
                    return lhs < rhs - 1e-12
            # 其他布尔组合，尝试数值化再转 bool（注意 SymPy Relational 不可直接 bool）
            val = cond.subs(opt_vars)
            try:
                return bool(val)
            except Exception:
                # 尝试数值化
                if hasattr(val, 'evalf'):
                    try:
                        return bool(val.evalf())
                    except Exception:
                        return False
                return False
        except Exception:
            return False

    def _sympy_subs_map(self, ir_con, opt_vars: Dict[str, float]) -> Dict:
        """将基于名称的 opt_vars 转换为 SymPy 符号的替换映射（仅覆盖该约束相关符号）。"""
        subs = {}
        try:
            for sym in getattr(ir_con, 'free_symbols', []) or []:
                name = str(sym)
                if name in opt_vars:
                    subs[sym] = opt_vars[name]
        except Exception:
            pass
        return subs

    def _aggregate_functional_result(self, result, agg: str) -> float:
        """按指定聚合方式将结果转为标量"""
        arr = np.asarray(result, dtype=float)
        if arr.size == 0:
            raise RuntimeError("FUNCTIONAL 约束评估为空，无法聚合")
        if agg == "l2_norm":
            return float(np.sqrt(np.sum(arr ** 2)))
        if agg == "l1_norm":
            return float(np.sum(np.abs(arr)))
        if agg == "max_abs":
            return float(np.max(np.abs(arr)))
        if agg == "mean_abs":
            return float(np.mean(np.abs(arr)))
        if agg == "final_state":
            return float(arr.ravel()[-1])
        raise RuntimeError(f"未知的 FUNCTIONAL 约束聚合方式: {agg}")

    def _coerce_to_scalar(self, result, identifier: str) -> float:
        """尝试将结果转为标量，必要时抛出可读错误"""
        if np.isscalar(result):
            return float(result)
        arr = np.asarray(result)
        if arr.size != 1:
            raise RuntimeError(
                f"FUNCTIONAL 约束 {identifier} 的评估结果为形状 {arr.shape}，缺少标量化策略；"
                f"请在 metadata.aggregation 中提供聚合方式或让 evaluator 返回标量"
            )
        return float(arr.reshape(-1)[0])

    # === 辅助方法 ===

    def _resolve_constraint_epsilon(self, ir_constraint) -> float:
        """确定严格不等式的epsilon"""
        if ir_constraint.epsilon_hint is not None:
            try:
                return float(ir_constraint.epsilon_hint)
            except Exception:
                pass
        return self.epsilon

    def _init_soft_eq_state(self, n_var: int) -> Dict[str, float]:
        """初始化软等式容差调度器"""
        initial = 1e-2 if n_var <= 10 else 5e-3
        return {
            'initial': initial,
            'min': 1e-6,
            'decay': 0.5,
            'step': max(50, 10 * max(1, n_var)),
            'current': initial,
            'evals': 0
        }

    def _update_soft_eq_state(self, state: Dict[str, float], batch_size: int) -> float:
        """根据评估次数动态收缩软等式阈值"""
        state['evals'] += int(batch_size)
        step = max(1, int(state.get('step', 100)))
        stage = state['evals'] // step
        current = state['initial'] * (state['decay'] ** stage)
        state['current'] = max(state['min'], current)
        return state['current']

    def _build_linear_equality_projection(self, equality_constraints, symbol_order):
        """提取线性等式用于快速投影"""
        if not equality_constraints:
            return None
        rows = []
        rhs = []
        symbols = symbol_order
        for ir_con in equality_constraints:
            expr = ir_con.normalized_expr
            if expr is None:
                continue
            try:
                A_row, b_row = sp.linear_eq_to_matrix([expr], symbols)
            except Exception:
                continue
            if A_row.shape[0] != 1:
                continue
            row = np.array(A_row, dtype=float).reshape(-1)
            if not np.any(np.abs(row) > 1e-10):
                continue
            rows.append(row)
            rhs.append(float(b_row[0]))
        if not rows:
            return None
        A = np.vstack(rows)
        b = np.array(rhs, dtype=float).reshape(-1, 1)
        try:
            gram = A @ A.T
            gram_inv = np.linalg.pinv(gram)
        except Exception:
            return None
        return {
            'A': A,
            'b': b,
            'At': A.T,
            'gram_inv': gram_inv
        }

    def _build_nonlinear_eq_specs(self, equality_constraints, symbol_to_index):
        """为非线性等式准备梯度近似"""
        specs = []
        for ir_con in equality_constraints:
            expr = ir_con.normalized_expr
            if expr is None or not ir_con.free_symbols:
                continue
            if self._is_linear_expression(expr):
                continue
            if ir_con.lambda_func is None:
                continue
            free_syms = list(ir_con.free_symbols)
            try:
                grad_funcs = [
                    sp.lambdify(free_syms, sp.diff(expr, sym), 'numpy')
                    for sym in free_syms
                ]
            except Exception:
                continue
            indices = [symbol_to_index[sym] for sym in free_syms if sym in symbol_to_index]
            if len(indices) != len(free_syms):
                continue
            specs.append({
                'lambda_func': ir_con.lambda_func,
                'grad_funcs': grad_funcs,
                'free_symbols': free_syms,
                'indices': indices
            })
        return specs

    # 将内部数据映射暴露给问题实例，便于结果解码
    # 注意：在 convert() 内部类 __init__ 中我们会设置这些属性

    def _build_opt_vars(self, row_values: np.ndarray, data: Dict[str, Any]) -> Dict[str, float]:
        """构建变量名到数值的映射"""
        symbol_order = data['symbol_order']
        # 仅基于 IR 变量顺序构建映射，避免将仿真自变量（如 t）视为决策变量
        return {str(sym): row_values[i] for i, sym in enumerate(symbol_order)}

    def _is_linear_expression(self, expr) -> bool:
        """粗略检测表达式是否线性"""
        try:
            for sym in expr.free_symbols:
                if expr.diff(sym, 2) != 0:
                    return False
            return True
        except Exception:
            return False

    def _repair_decision_vectors(self, X: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
        """在评估前修复候选解，使其更易满足等式/严格不等式。"""
        if X is None:
            return X
        arr = np.array(X, dtype=float, copy=True)
        original_1d = arr.ndim == 1
        if original_1d:
            arr = arr.reshape(1, -1)
        soft_state = data.get('soft_eq_state') or {}
        current_tol = soft_state.get('current', 1e-3)
        # 盒投影（前）
        arr = np.clip(arr, data['xl'], data['xu'])
        projection = data.get('linear_eq_projection')
        if projection:
            arr = self._apply_linear_projection(arr, projection, data)
        # 严格不等式的内点化修复（可选）
        if self.strict_ieq_enable:
            arr = self._apply_strict_ineq_interior_push(arr, data, current_tol)
        # 多步非线性等式修复
        arr = self._apply_nonlinear_eq_correction(arr, data, current_tol)
        # 盒投影（后）
        arr = np.clip(arr, data['xl'], data['xu'])
        if original_1d:
            return arr[0]
        return arr

    def _apply_linear_projection(self, arr: np.ndarray, projection: Dict[str, Any], data: Dict[str, Any]) -> np.ndarray:
        """将解投影到线性等式子空间。"""
        try:
            A = projection['A']
            b = projection['b']
            At = projection['At']
            gram_inv = projection['gram_inv']
        except KeyError:
            return arr
        for i in range(arr.shape[0]):
            row = arr[i].reshape(-1, 1)
            residual = (A @ row) - b
            correction = At @ (gram_inv @ residual)
            arr[i] = (row - correction).reshape(-1)
        return np.clip(arr, data['xl'], data['xu'])

    def _apply_nonlinear_eq_correction(self, arr: np.ndarray, data: Dict[str, Any], tolerance: float) -> np.ndarray:
        """使用阻尼牛顿+线搜索/信赖域多步修复非线性等式（方案1）。"""
        specs = data.get('nonlinear_eq_specs') or []
        if not specs:
            # 即使代数等式为空，仍可能存在 FUNCTIONAL 等式，需要专门处理
            pass
        n_var = arr.shape[1]
        # 信赖域半径：按变量跨度与维度自适应
        ranges = np.maximum(data['xu'] - data['xl'], 1e-12)
        base_radius = float(np.linalg.norm(ranges) / max(np.sqrt(max(1, n_var)), 1.0))
        trust_radius = float(self.repair_trust_radius_factor * base_radius) if self.repair_trust_radius_factor > 0 else None
        for i in range(arr.shape[0]):
            row = arr[i]
            # 多轮修复迭代
            for _ in range(int(max(1, self.repair_max_steps))):
                improved = False
                # 1) 代数等式修复（若存在）
                for spec in specs:
                    free_vals = [row[idx] for idx in spec['indices']]
                    try:
                        value = float(spec['lambda_func'](*free_vals))
                    except Exception:
                        continue
                    if abs(value) <= tolerance:
                        continue
                    grad = np.zeros(n_var, dtype=float)
                    for grad_func, idx in zip(spec['grad_funcs'], spec['indices']):
                        try:
                            grad[idx] = float(grad_func(*free_vals))
                        except Exception:
                            grad[idx] = 0.0
                    gnorm = np.linalg.norm(grad)
                    if gnorm < 1e-10:
                        continue
                    d = (value / (gnorm ** 2)) * grad
                    if trust_radius is not None:
                        dnorm = float(np.linalg.norm(d))
                        if dnorm > trust_radius:
                            d = d * (trust_radius / dnorm)
                    alpha = 1.0
                    if self.repair_linesearch:
                        for _ls in range(int(max(1, self.repair_linesearch_max_steps))):
                            trial = row - alpha * d
                            trial = np.clip(trial, data['xl'], data['xu'])
                            tvals = [trial[idx] for idx in spec['indices']]
                            try:
                                new_val = float(spec['lambda_func'](*tvals))
                            except Exception:
                                new_val = value
                            if abs(new_val) <= (1.0 - self.repair_linesearch_c * alpha) * abs(value):
                                break
                            alpha *= 0.5
                    new_row = row - alpha * d
                    new_row = np.clip(new_row, data['xl'], data['xu'])
                    try:
                        nvals = [new_row[idx] for idx in spec['indices']]
                        new_val2 = float(spec['lambda_func'](*nvals))
                    except Exception:
                        new_val2 = value
                    if abs(new_val2) < abs(value):
                        row[:] = new_row
                        improved = True

                # 2) FUNCTIONAL 等式修复（数值梯度）
                functional_eqs = [c for c in data.get('equality_constraints', []) if getattr(c, 'category', None) == IRConstraintCategory.FUNCTIONAL]
                if functional_eqs:
                    # 变量跨度用于差分步长
                    spans = np.maximum(data['xu'] - data['xl'], 1e-12)
                    for ir_con in functional_eqs:
                        # 当前等式值
                        opt_vars = self._build_opt_vars(row, data)
                        try:
                            value = float(self._evaluate_functional_constraint(ir_con, opt_vars))
                        except Exception:
                            continue
                        if abs(value) <= tolerance:
                            continue
                        # 数值梯度（对所有决策变量）
                        grad = np.zeros(n_var, dtype=float)
                        for j in range(n_var):
                            h = 1e-6 * spans[j]
                            if h <= 0:
                                h = 1e-6
                            row[j] += h
                            opt1 = self._build_opt_vars(np.clip(row, data['xl'], data['xu']), data)
                            try:
                                ev1 = float(self._evaluate_functional_constraint(ir_con, opt1))
                            except Exception:
                                ev1 = value
                            row[j] -= 2*h
                            opt2 = self._build_opt_vars(np.clip(row, data['xl'], data['xu']), data)
                            try:
                                ev2 = float(self._evaluate_functional_constraint(ir_con, opt2))
                            except Exception:
                                ev2 = value
                            row[j] += h
                            grad[j] = (ev1 - ev2) / (2*h)
                        gnorm = np.linalg.norm(grad)
                        if gnorm < 1e-10:
                            continue
                        d = (value / (gnorm ** 2)) * grad
                        if trust_radius is not None:
                            dnorm = float(np.linalg.norm(d))
                            if dnorm > trust_radius:
                                d = d * (trust_radius / dnorm)
                        alpha = 1.0
                        if self.repair_linesearch:
                            for _ls in range(int(max(1, self.repair_linesearch_max_steps))):
                                trial = row - alpha * d
                                trial = np.clip(trial, data['xl'], data['xu'])
                                opt_t = self._build_opt_vars(trial, data)
                                try:
                                    new_val = float(self._evaluate_functional_constraint(ir_con, opt_t))
                                except Exception:
                                    new_val = value
                                if abs(new_val) <= (1.0 - self.repair_linesearch_c * alpha) * abs(value):
                                    break
                                alpha *= 0.5
                        new_row = row - alpha * d
                        new_row = np.clip(new_row, data['xl'], data['xu'])
                        opt_n = self._build_opt_vars(new_row, data)
                        try:
                            new_val2 = float(self._evaluate_functional_constraint(ir_con, opt_n))
                        except Exception:
                            new_val2 = value
                        if abs(new_val2) < abs(value):
                            row[:] = new_row
                            improved = True
                if not improved:
                    break
        return np.clip(arr, data['xl'], data['xu'])

    def _apply_strict_ineq_interior_push(self, arr: np.ndarray, data: Dict[str, Any], tolerance: float) -> np.ndarray:
        """对严格不等式执行内点化小步（方案2）。"""
        ineqs = [con for con in data['inequality_constraints'] if getattr(con, 'strict', False)]
        if not ineqs:
            return arr
        xl, xu = data['xl'], data['xu']
        spans = np.maximum(xu - xl, 1e-12)
        step_base = float(self.strict_ieq_step_factor * np.min(spans))
        for i in range(arr.shape[0]):
            row = arr[i].copy()
            # 计算每条严格不等式的违反度，并选择前k条
            viols = []
            for ir_con in ineqs:
                expr = ir_con.normalized_expr
                # 评估expr_val
                if ir_con.lambda_func is not None:
                    args = [row[data['symbol_to_index'][sym]] for sym in ir_con.free_symbols]
                    expr_val = float(ir_con.lambda_func(*args))
                else:
                    substitutions = {sym: row[data['symbol_to_index'][sym]] for sym in ir_con.free_symbols}
                    expr_val = float(expr.subs(substitutions).evalf())
                eps = self._resolve_constraint_epsilon(ir_con)
                if ir_con.sense == IRConstraintSense.LE:
                    g = expr_val + eps
                else:
                    g = -expr_val + eps
                if g > 0:
                    viols.append((g, ir_con))
            if not viols:
                continue
            viols.sort(key=lambda t: t[0], reverse=True)
            for g, ir_con in viols[:max(1, self.strict_ieq_topk)]:
                # 数值梯度（对涉及的自由变量做中心差分）
                grad = np.zeros(arr.shape[1], dtype=float)
                indices = [data['symbol_to_index'][sym] for sym in ir_con.free_symbols]
                for idx in indices:
                    h = 1e-6 * spans[idx]
                    if h <= 0:
                        h = 1e-6
                    row[idx] += h
                    ev1 = self._eval_constraint_scalar(ir_con, row, data)
                    row[idx] -= 2*h
                    ev2 = self._eval_constraint_scalar(ir_con, row, data)
                    row[idx] += h
                    grad[idx] = (ev1 - ev2) / (2*h)
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-10:
                    continue
                # 推内方向（使g减少）：对于LE约束，g=expr+eps，沿 -grad；GE约束对应 expr→-expr 已统一成 g
                d = (g / gnorm) * (grad / gnorm)
                new_row = row - step_base * d
                row = np.clip(new_row, xl, xu)
            arr[i] = row
        return np.clip(arr, data['xl'], data['xu'])

    def _eval_constraint_scalar(self, ir_con, row: np.ndarray, data: Dict[str, Any]) -> float:
        if ir_con.lambda_func is not None:
            args = [row[data['symbol_to_index'][sym]] for sym in ir_con.free_symbols]
            return float(ir_con.lambda_func(*args))
        expr = ir_con.normalized_expr
        substitutions = {sym: row[data['symbol_to_index'][sym]] for sym in ir_con.free_symbols}
        return float(expr.subs(substitutions).evalf())
