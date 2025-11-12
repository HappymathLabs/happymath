"""
Pyomo适配器

将ParseResult转换为Pyomo模型。
在原有稳定性的基础上，引入两类增强：
1) 分段线性自动化：尝试把单变量分段线性表达式（含断点+线性段）建模为Pyomo Piecewise（支持SOS2/凸组合），
   适用于目标与约束的任意位置；对无法线性化的分段表达式保留原策略（在目标中拒绝，在约束中转GDP）。
2) GDP强化：在Big-M路径上按约束级估计M（线性表达式的上下界），优先使用Hull（变量有界时），
   超过分支阈值自动回退到Big-M；同时暴露诊断信息，便于调参与问题定位。
"""

import pyomo.environ as pyo
from pyomo.gdp import Disjunct, Disjunction
import pyomo.gdp.plugins  # 注册GDP相关变换（包含 gdp.hull）
from pyomo.core import TransformationFactory
import numpy as np
from sympy import Eq, Ge, Gt, Le, Lt, Symbol, Number
from sympy import Add, Mul, Pow, nsimplify
from sympy import Rational
from sympy import ilcm
from sympy import sin, cos, tan, exp, log, sqrt, Abs
from sympy import Piecewise, floor, ceiling, sign
from ..ir import IRConstraintCategory, IRConstraintSense, IROptVarType

from ..opt_core.opt_exceptions import ConversionError


class PyomoAdapter:
    """Pyomo模型适配器"""

    def __init__(self, parse_result, epsilon=1e-6, gdp_strategy: str = "auto", piecewise_strategy: str = "auto", gdp_max_branches: int = 40, bigm_mode: str = "per_constraint"):
        """
        初始化Pyomo适配器

        Args:
            parse_result: ParseResult实例
            epsilon: 默认epsilon值
            gdp_strategy: GDP 处理策略（auto/hull/bigm）
            piecewise_strategy: 分段线性建模策略（auto/sos2/cc）
            gdp_max_branches: GDP分支数阈值，超过后更倾向Big-M
            bigm_mode: Big-M估计模式（per_constraint/global）
        """
        self.parse_result = parse_result
        self.epsilon = float(epsilon)
        self.gdp_strategy = gdp_strategy.lower() if isinstance(gdp_strategy, str) else "auto"
        self.piecewise_strategy = (piecewise_strategy or "auto").lower()
        self.gdp_max_branches = int(gdp_max_branches or 40)
        self.bigm_mode = (bigm_mode or "per_constraint").lower()
        # 运行时持有当前Pyomo模型句柄（用于创建Piecewise等组件）
        self._model = None
        # GDP Big-M逐约束映射（在构建disjunct约束时填充）
        self._bigm_map = {}

    def convert(self):
        """
        将ParseResult转换为Pyomo模型

        Returns:
            pyo.ConcreteModel: Pyomo模型
        """
        model = pyo.ConcreteModel()
        # 保持模型句柄供后续表达式转换使用（例如创建Piecewise组件）
        self._model = model

        # 仅支持标准优化问题（LOGICAL 约束不在 Pyomo 路径处理）
        return self._convert_standard_optimization(model)

    # 仿真优化路径已移除

    def _convert_standard_optimization(self, model):
        """转换标准优化问题"""
        # 1. 定义变量
        self._add_variables(model)

        # 2. 定义目标函数
        self._add_objectives(model)

        # 3. 定义约束
        self._add_constraints(model)

        return model



    def _add_variables(self, model):
        """添加变量定义"""
        ir_problem = self.parse_result.ir_problem
        sympy_to_pyomo_var_map = {}

        for ir_var in ir_problem.variables:
            name = ir_var.name
            lb = self._normalize_bound(ir_var.lower_bound)
            ub = self._normalize_bound(ir_var.upper_bound)

            if ir_var.var_type == IROptVarType.BINARY:
                var_obj = pyo.Var(within=pyo.Binary, name=name)
            elif ir_var.var_type == IROptVarType.INTEGER:
                var_obj = pyo.Var(within=pyo.Integers, bounds=(lb, ub), name=name)
            elif ir_var.var_type == IROptVarType.ENUM:
                var_obj = self._create_enumeration_variable(model, ir_var, lb, ub)
            else:
                var_obj = pyo.Var(within=pyo.Reals, bounds=(lb, ub), name=name)

            setattr(model, name, var_obj)
            sympy_to_pyomo_var_map[ir_var.symbol] = var_obj

        model._sympy_to_pyomo_var_map = sympy_to_pyomo_var_map

    def _normalize_bound(self, value):
        """将边界值转换为Pyomo可接受的浮点数"""
        if value is None:
            return None
        try:
            if value in (np.inf, -np.inf):
                return None
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return None

    def _create_enumeration_variable(self, model, ir_var, lb, ub):
        """为枚举变量创建Pyomo变量及选择器约束"""
        domain = ir_var.discrete_domain.values if ir_var.discrete_domain else ()
        if not domain:
            return pyo.Var(within=pyo.Reals, bounds=(lb, ub), name=ir_var.name)

        numeric_values = []
        seen = set()
        for v in domain:
            try:
                num = float(v)
            except Exception as exc:
                raise ConversionError(
                    target_format='pyomo',
                    expression=v,
                    message=f'离散域值无法转换为浮点数: {v} ({exc})'
                ) from exc
            if num not in seen:
                numeric_values.append(num)
                seen.add(num)

        if not numeric_values:
            return pyo.Var(within=pyo.Reals, bounds=(lb, ub), name=ir_var.name)

        unique_sorted = sorted(numeric_values)

        # 若离散域是标准{0,1}，直接返回Binary变量
        if set(unique_sorted) == {0.0, 1.0}:
            return pyo.Var(within=pyo.Binary, name=ir_var.name)

        # 若离散域为连续整数区间，则可以建模为整数变量
        if all(float(v).is_integer() for v in unique_sorted):
            ints = [int(round(v)) for v in unique_sorted]
            ints_sorted = sorted(set(ints))
            if ints_sorted == list(range(ints_sorted[0], ints_sorted[-1] + 1)):
                return pyo.Var(
                    within=pyo.Integers,
                    bounds=(ints_sorted[0], ints_sorted[-1]),
                    name=ir_var.name
                )

        lo = min(unique_sorted) if lb is None else lb
        hi = max(unique_sorted) if ub is None else ub
        var = pyo.Var(within=pyo.Reals, bounds=(lo, hi), name=ir_var.name)
        index = pyo.RangeSet(len(unique_sorted))
        selector = pyo.Var(index, within=pyo.Binary)
        selector_name = f"{ir_var.name}__enum_sel"
        model.add_component(selector_name, selector)
        model.add_component(
            f"{ir_var.name}__enum_sum1",
            pyo.Constraint(expr=sum(selector[i] for i in index) == 1)
        )
        model.add_component(
            f"{ir_var.name}__enum_link",
            pyo.Constraint(expr=var == sum(unique_sorted[i - 1] * selector[i] for i in index))
        )

        if not hasattr(model, "_enum_metadata"):
            model._enum_metadata = {}
            model._enum_metadata[ir_var.symbol] = {
                'values': unique_sorted,
                'selector': selector
            }
        return var

    def _normalize_objective_expr(self, expr):
        """对目标表达式做预处理，尽量保持有理数系数"""
        processed = expr
        if self._is_linear_expr(processed):
            try:
                processed = nsimplify(processed, rational=True)
                processed = self._scale_linear_expr_to_integer(processed)
            except Exception:
                return processed
        return processed

    def _normalize_constraint_expr(self, expr):
        """约束表达式的标准化处理"""
        processed = expr
        if self._is_linear_expr(processed):
            try:
                processed = nsimplify(processed, rational=True)
            except Exception:
                return processed
        return processed

    def _resolve_epsilon(self, ir_constraint):
        """确定严格不等式使用的epsilon"""
        if ir_constraint.epsilon_hint is not None:
            try:
                return float(ir_constraint.epsilon_hint)
            except Exception:
                return self.epsilon
        return self.epsilon

    def _add_objectives(self, model):
        """添加目标函数定义"""
        ir_problem = self.parse_result.ir_problem
        sympy_to_pyomo_var_map = model._sympy_to_pyomo_var_map
        objectives = ir_problem.objectives

        if any(obj.is_functional for obj in objectives):
            raise ConversionError(
                target_format='pyomo',
                expression=None,
                message='当前Pyomo适配器不直接支持函数式目标，请选择Pymoo或先将目标离散化。'
            )

        def conv(e):
            return self._sympy_to_pyomo_expr(e, sympy_to_pyomo_var_map)

        if len(objectives) == 1:
            expr_sym = self._normalize_objective_expr(objectives[0].expression)
            expr = conv(expr_sym)
            sense = pyo.minimize if objectives[0].sense == 'min' else pyo.maximize
            model.obj = pyo.Objective(expr=expr, sense=sense)
        else:
            model.obj_list = pyo.ObjectiveList()
            for obj in objectives:
                expr_sym = self._normalize_objective_expr(obj.expression)
                expr = conv(expr_sym)
                sense = pyo.minimize if obj.sense == 'min' else pyo.maximize
                model.obj_list.add(expr, sense=sense)

    def _add_constraints(self, model):
        """添加约束定义"""
        ir_problem = self.parse_result.ir_problem
        sympy_to_pyomo_var_map = model._sympy_to_pyomo_var_map
        model.con_list = pyo.ConstraintList()

        for ir_con in ir_problem.constraints:
            if ir_con.category == IRConstraintCategory.DOMAIN:
                # 已通过变量域处理
                continue
            if ir_con.category == IRConstraintCategory.FUNCTIONAL:
                raise ConversionError(
                    target_format='pyomo',
                    expression=ir_con.original,
                    message='当前Pyomo适配器暂不支持功能型约束，请选择Pymoo或提供离散化策略。'
                )
            if ir_con.category == IRConstraintCategory.LOGICAL:
                # 将逻辑约束（Piecewise/指示约束）转录为 GDP，并使用 HULL 转换
                self._add_logical_constraint(model, ir_con, sympy_to_pyomo_var_map)
                continue
            if ir_con.category != IRConstraintCategory.ALGEBRAIC:
                continue

            expr = ir_con.normalized_expr
            if expr is None and ir_con.lhs is not None and ir_con.rhs is not None:
                expr = (ir_con.lhs - ir_con.rhs).expand()
            if expr is None:
                continue

            expr = self._normalize_constraint_expr(expr)
            body = self._sympy_to_pyomo_expr(expr, sympy_to_pyomo_var_map)
            epsilon = self._resolve_epsilon(ir_con)

            if ir_con.sense == IRConstraintSense.EQ:
                model.con_list.add(body == 0)
            elif ir_con.sense == IRConstraintSense.LE:
                rhs = -epsilon if ir_con.strict else 0
                model.con_list.add(body <= rhs)
            elif ir_con.sense == IRConstraintSense.GE:
                rhs = epsilon if ir_con.strict else 0
                model.con_list.add(body >= rhs)

        self._apply_gdp_reformulation(model)
        return model

    def _add_logical_constraint(self, model, ir_con, var_map):
        """将 LOGICAL 约束（Piecewise/Indicator）转录为 GDP 并挂到模型上。

        仅支持 metadata 中 'logical_kind' == 'piecewise'，branches 为 [(expr, cond)].
        expr/cond 允许 Eq/Ge/Le/Gt/Lt/True/False。
        """
        meta = ir_con.metadata or {}
        kind = meta.get('logical_kind')
        if kind != 'piecewise':
            raise ConversionError(target_format='pyomo', expression=ir_con.original,
                                  message=f'未支持的LOGICAL约束类型: {kind}')

        branches = meta.get('branches') or []
        if not branches:
            raise ConversionError(target_format='pyomo', expression=ir_con.original,
                                  message='Piecewise 约束缺少分支信息')

        # 为该逻辑约束创建一组 Disjuncts
        disjuncts = []
        base_name = ir_con.identifier
        for i, br in enumerate(branches):
            dj_name = f"{base_name}_dj{i}"
            dj = Disjunct()
            model.add_component(dj_name, dj)

            expr = br.get('expr')
            cond = br.get('cond')

            # 在 disjunct 中添加条件约束
            self._add_conditional_constraint_to_disjunct(dj, cond, var_map)
            # 在 disjunct 中添加分支表达式约束
            self._add_branch_expr_to_disjunct(dj, expr, var_map)

            disjuncts.append(dj)

        # 添加 Disjunction（精确分支选择）
        djcomp_name = f"{base_name}_disj"
        # 注意：Disjunction 需要传入 Disjunct 对象列表
        model.add_component(djcomp_name, Disjunction(expr=disjuncts))

    def _add_conditional_constraint_to_disjunct(self, disjunct, cond, var_map):
        """将分支条件作为约束放入 disjunct；True/False 特殊处理。"""
        from sympy import Eq, Ge, Le, Gt, Lt, S
        if cond is True or cond is S.true or cond is None:
            return
        if cond is False or cond is S.false:
            # 添加一个恒不等式使该分支在被选中时不可行
            cname = f"{disjunct.name}_cond_false"
            disjunct.add_component(cname, pyo.Constraint(expr=(1 <= 0)))
            return
        # 关系表达式
        # 将 SymPy 关系表达式转换为 Pyomo Constraint
        self._add_relational_to_block(disjunct, cond, var_map, suffix='_cond')

    def _add_branch_expr_to_disjunct(self, disjunct, expr, var_map):
        from sympy import Eq, Ge, Le, Gt, Lt, S
        if expr is True or expr is S.true or expr is None:
            return
        if expr is False or expr is S.false:
            cname = f"{disjunct.name}_expr_false"
            disjunct.add_component(cname, pyo.Constraint(expr=(1 <= 0)))
            return
        self._add_relational_to_block(disjunct, expr, var_map, suffix='_expr')

    def _add_relational_to_block(self, block, rel, var_map, suffix=''):
        """将 SymPy 关系表达式（Eq/Ge/Le/Gt/Lt）添加为给定块内的约束。
        对于严格不等式（Gt/Lt），使用适配器的 epsilon 进行近似。
        """
        from sympy import Eq, Ge, Le, Gt, Lt
        epsilon = self.epsilon
        if isinstance(rel, Eq):
            body_sym = rel.lhs - rel.rhs
            body = self._sympy_to_pyomo_expr(body_sym, var_map)
            cname = f"rel_eq{suffix}"
            con = pyo.Constraint(expr=(body == 0))
            block.add_component(cname, con)
            self._record_bigm_for_constraint(con, body_sym, sense='eq')
        elif isinstance(rel, Le):
            body_sym = rel.lhs - rel.rhs
            body = self._sympy_to_pyomo_expr(body_sym, var_map)
            cname = f"rel_le{suffix}"
            con = pyo.Constraint(expr=(body <= 0))
            block.add_component(cname, con)
            self._record_bigm_for_constraint(con, body_sym, sense='le')
        elif isinstance(rel, Ge):
            body_sym = rel.lhs - rel.rhs
            body = self._sympy_to_pyomo_expr(body_sym, var_map)
            cname = f"rel_ge{suffix}"
            con = pyo.Constraint(expr=(body >= 0))
            block.add_component(cname, con)
            self._record_bigm_for_constraint(con, body_sym, sense='ge')
        elif isinstance(rel, Gt):
            body_sym = rel.lhs - rel.rhs + epsilon
            body = self._sympy_to_pyomo_expr(body_sym, var_map)
            cname = f"rel_gt{suffix}"
            con = pyo.Constraint(expr=(body >= 0))
            block.add_component(cname, con)
            self._record_bigm_for_constraint(con, rel.lhs - rel.rhs, sense='ge')
        elif isinstance(rel, Lt):
            body_sym = rel.lhs - rel.rhs - epsilon
            body = self._sympy_to_pyomo_expr(body_sym, var_map)
            cname = f"rel_lt{suffix}"
            con = pyo.Constraint(expr=(body <= 0))
            block.add_component(cname, con)
            self._record_bigm_for_constraint(con, rel.lhs - rel.rhs, sense='le')
        else:
            raise ConversionError(target_format='pyomo', expression=rel, message='不支持的关系表达式类型')

    # === SymPy → Pyomo 表达式转换 ===
    def _sympy_to_pyomo_expr(self, expr, sympy_to_pyomo_var_map):
        """将 SymPy 表达式转换为 Pyomo 表达式（递归）"""
        # 符号
        if isinstance(expr, Symbol):
            if expr not in sympy_to_pyomo_var_map:
                raise ConversionError(target_format='pyomo', expression=expr,
                                      message=f'未找到符号 {expr} 对应的Pyomo变量')
            return sympy_to_pyomo_var_map[expr]

        # 常数
        if isinstance(expr, Number):
            return float(expr)

        # 组合节点
        f = expr.func
        args = expr.args

        # 基本代数
        if f is Add:
            return sum(self._sympy_to_pyomo_expr(a, sympy_to_pyomo_var_map) for a in args)
        if f is Mul:
            prod = 1
            for a in args:
                prod = prod * self._sympy_to_pyomo_expr(a, sympy_to_pyomo_var_map)
            return prod
        if f is Pow:
            base = self._sympy_to_pyomo_expr(args[0], sympy_to_pyomo_var_map)
            expn = args[1]
            if isinstance(expn, Number):
                return base ** float(expn)
            else:
                expn_conv = self._sympy_to_pyomo_expr(expn, sympy_to_pyomo_var_map)
                return base ** expn_conv

        # 常见函数
        if f is Abs:
            return pyo.abs(self._sympy_to_pyomo_expr(args[0], sympy_to_pyomo_var_map))
        if f is sin:
            return pyo.sin(self._sympy_to_pyomo_expr(args[0], sympy_to_pyomo_var_map))
        if f is cos:
            return pyo.cos(self._sympy_to_pyomo_expr(args[0], sympy_to_pyomo_var_map))
        if f is tan:
            return pyo.tan(self._sympy_to_pyomo_expr(args[0], sympy_to_pyomo_var_map))
        if f is exp:
            return pyo.exp(self._sympy_to_pyomo_expr(args[0], sympy_to_pyomo_var_map))
        if f is log:
            return pyo.log(self._sympy_to_pyomo_expr(args[0], sympy_to_pyomo_var_map))
        if f is sqrt:
            return pyo.sqrt(self._sympy_to_pyomo_expr(args[0], sympy_to_pyomo_var_map))
        if f is floor:
            return pyo.floor(self._sympy_to_pyomo_expr(args[0], sympy_to_pyomo_var_map))
        if f is ceiling:
            return pyo.ceil(self._sympy_to_pyomo_expr(args[0], sympy_to_pyomo_var_map))
        if f is sign:
            arg_expr = self._sympy_to_pyomo_expr(args[0], sympy_to_pyomo_var_map)
            negative_branch = pyo.Expr_if(IF=(arg_expr, '<', 0), THEN=-1, ELSE=0)
            return pyo.Expr_if(IF=(arg_expr, '>', 0), THEN=1, ELSE=negative_branch)

        # 分段函数：尝试将单变量分段线性表达式转为Pyomo Piecewise（SOS2/CC）
        if f is Piecewise:
            pw_expr = self._try_convert_piecewise_linear(expr, sympy_to_pyomo_var_map)
            if pw_expr is not None:
                return pw_expr
            # 无法线性化时保留原策略（目标中拒绝，约束路径通过 LOGICAL→GDP 处理）
            raise ConversionError(
                target_format='pyomo',
                expression=expr,
                message='检测到非线性或多变量分段函数，当前版本仅自动处理单变量分段线性。' \
                        '请将其改写为线性分段或使用Pymoo；若为约束层面的Piecewise，可使用GDP路径。'
            )

        # 默认回退：尝试将常数化表达式直接转 float
        try:
            return float(expr)
        except Exception:
            raise ConversionError(target_format='pyomo', expression=expr,
                                  message=f'无法将表达式 {expr} 转换为Pyomo表达式（未支持的构造），可考虑改用 Pymoo 模式或提供等价的代数建模。')

    def _is_linear_expr(self, expr):
        """粗略检查表达式是否线性：对所有变量二阶导为0"""
        try:
            for s in self.parse_result.all_symbols:
                if expr.diff(s, 2) != 0:
                    return False
            return True
        except Exception:
            return False

    def _scale_linear_pair_to_integer(self, lin_expr, const_float):
        """
        将线性表达式 lin_expr 及常数 const 缩放为整数系数（乘以分母最小公倍数）。
        返回 (lin_expr_scaled, const_scaled_float)。
        """
        # 收集分母
        denoms = []
        try:
            for num in lin_expr.atoms(Number):
                rat = nsimplify(num, rational=True)
                if isinstance(rat, Rational):
                    denoms.append(rat.q)
        except Exception:
            pass
        try:
            ratc = nsimplify(const_float, rational=True)
            if isinstance(ratc, Rational):
                denoms.append(ratc.q)
        except Exception:
            pass
        if not denoms:
            return lin_expr, float(const_float)
        factor = 1
        for d in denoms:
            try:
                factor = ilcm(int(factor), int(d))
            except Exception:
                pass
        if factor == 1:
            return lin_expr, float(const_float)
        return (lin_expr * factor), float(const_float * factor)

    def _scale_linear_expr_to_integer(self, expr):
        """将线性目标表达式整体按分母最小公倍数放大为整数系数"""
        denoms = []
        try:
            for num in expr.atoms(Number):
                rat = nsimplify(num, rational=True)
                if isinstance(rat, Rational):
                    denoms.append(rat.q)
        except Exception:
            pass
        if not denoms:
            return expr
        factor = 1
        for d in denoms:
            try:
                factor = ilcm(int(factor), int(d))
            except Exception:
                pass
        if factor == 1:
            return expr
        return expr * factor

    # === GDP 处理 ===
    def _apply_gdp_reformulation(self, model):
        """根据策略选择 Hull 或 Big-M 处理 GDP。"""
        disjunctions = list(model.component_objects(Disjunction, active=True))
        if not disjunctions:
            return
        strategy = self._resolve_gdp_strategy(disjunctions)
        try:
            if strategy == 'hull':
                TransformationFactory('gdp.hull').apply_to(model)
            else:
                # Big-M：若已生成逐约束M映射则优先使用
                if self.bigm_mode == 'per_constraint' and self._bigm_map:
                    model._bigm_map = dict(self._bigm_map)
                    TransformationFactory('gdp.bigm').apply_to(model, bigM=model._bigm_map)
                else:
                    big_m_value = self._estimate_big_m()
                    TransformationFactory('gdp.bigm').apply_to(model, bigM=big_m_value)
        except Exception as exc:
            raise ConversionError(
                target_format='pyomo',
                expression=None,
                message=(
                    f"GDP 转换（策略: {strategy}）失败: {exc}. "
                    "建议减少分支数量或改用 Pymoo 启发式求解。"
                )
            ) from exc

    def _resolve_gdp_strategy(self, disjunctions) -> str:
        """在 auto 模式下按分支规模选择策略。"""
        if self.gdp_strategy in {'hull', 'bigm'}:
            return self.gdp_strategy
        bound_manager = getattr(self.parse_result, 'bound_manager', None)
        if bound_manager is not None and not bound_manager.check_all_variables_bounded():
            return 'bigm'
        total_branches = sum(len(list(disj.disjuncts)) for disj in disjunctions)
        if total_branches >= int(self.gdp_max_branches):
            return 'bigm'
        return 'hull'

    def _estimate_big_m(self) -> float:
        """基于真实边界估计 Big-M。"""
        bound_manager = getattr(self.parse_result, 'bound_manager', None)
        if bound_manager is None:
            return 1e4
        try:
            lower = np.array(bound_manager.lower_bounds, dtype=float, copy=True)
            upper = np.array(bound_manager.upper_bounds, dtype=float, copy=True)
        except Exception:
            return 1e4
        default_range = float(getattr(bound_manager, 'default_search_range', 100.0) or 100.0)
        lower = np.where(np.isfinite(lower), lower, -default_range)
        upper = np.where(np.isfinite(upper), upper, default_range)
        spans = np.abs(upper - lower)
        magnitudes = np.maximum(np.abs(lower), np.abs(upper))
        span_sum = float(np.sum(spans)) if spans.size else 1.0
        magnitude_max = float(np.max(magnitudes)) if magnitudes.size else 1.0
        baseline = max(span_sum, magnitude_max)
        return max(10.0, baseline * 1.2)

    # === 分段线性：SymPy Piecewise → Pyomo Piecewise ===
    def _try_convert_piecewise_linear(self, expr, var_map):
        """若表达式为单变量分段线性，构建Pyomo Piecewise并返回其表示变量；否则返回None。

        当前仅支持：
        - 自变量仅一个，且在 var_map 中；
        - 每段表达式关于该自变量为一次函数；
        - 断点来自条件中的常数，以变量边界作为封闭区间端点；
        """
        try:
            # 仅支持单变量
            free_syms = list(expr.free_symbols)
            if len(free_syms) != 1:
                return None
            x = free_syms[0]
            if x not in var_map:
                return None
            # 变量需要有限边界
            bm = getattr(self.parse_result, 'bound_manager', None)
            if bm is None:
                return None
            sym2idx = self.parse_result.symbol_to_index
            idx = sym2idx.get(x)
            if idx is None:
                return None
            xl = float(bm.lower_bounds[idx])
            xu = float(bm.upper_bounds[idx])
            if not (np.isfinite(xl) and np.isfinite(xu)):
                return None

            # 收集断点
            pts = set([xl, xu])
            for (piece, cond) in expr.as_expr_set_pairs():
                # 仅接受线性段
                if not self._is_affine_in_symbol(piece, x):
                    return None
                # 条件中提取常数阈值
                try:
                    from sympy import Relational
                    if cond is True:
                        continue
                    if isinstance(cond, Relational):
                        for side in (cond.lhs, cond.rhs):
                            val = self._to_float_if_const(side)
                            if val is not None:
                                pts.add(val)
                except Exception:
                    pass
            pw_pts = sorted([p for p in pts if xl <= p <= xu])
            if len(pw_pts) < 2:
                pw_pts = [xl, xu]

            # 计算每个断点的函数值
            f_vals = []
            for v in pw_pts:
                try:
                    f_vals.append(float(expr.subs({x: v}).evalf()))
                except Exception:
                    return None

            # 创建一个新的变量承接分段线性函数值
            y_name = f"_pw_{str(x)}_{len(list(getattr(self._model, 'component_objects', lambda *_: [])()))}"
            y_var = pyo.Var(initialize=f_vals[0])
            self._model.add_component(y_name, y_var)

            # 选择表示（SOS2 或 convex combination）
            repn = 'SOS2' if self.piecewise_strategy in {'auto', 'sos2'} else 'CC'
            # 使用Pyomo核心的Piecewise组件
            # 注意：Piecewise会根据pw_pts与f_rule在模型中添加必要的变量与约束
            pw_name = f"{y_name}__pw"
            self._model.add_component(
                pw_name,
                pyo.Piecewise(
                    y_var,
                    var_map[x],
                    pw_pts=pw_pts,
                    f_rule=f_vals,
                    pw_constr_type='EQ',
                    pw_repn=repn
                )
            )
            return y_var
        except Exception:
            return None

    def _is_affine_in_symbol(self, expr, sym):
        """判断expr是否关于指定符号为一次（仿射）。"""
        try:
            poly = None
            try:
                import sympy as sp
                poly = sp.Poly(sp.expand(expr), sym)
            except Exception:
                poly = None
            if poly is None:
                # 退化到求二阶导判断
                return expr.diff(sym, 2) == 0
            return poly.total_degree() <= 1
        except Exception:
            return False

    def _to_float_if_const(self, expr):
        try:
            if not hasattr(expr, 'free_symbols') or len(expr.free_symbols) == 0:
                return float(expr)
        except Exception:
            pass
        return None

    # === Big-M逐约束估计（线性表达式） ===
    def _record_bigm_for_constraint(self, con, body_sym_expr, sense: str):
        """为disjunct内新增的约束记录Big-M估计（仅线性表达式）。"""
        try:
            M = self._estimate_M_for_linear(body_sym_expr, sense)
            if M is not None and np.isfinite(M):
                self._bigm_map[con] = float(max(0.0, M))
        except Exception:
            pass

    def _estimate_M_for_linear(self, body_sym_expr, sense: str):
        """给定标准化的body(x)（<=0 或 >=0 的左端），用变量边界估计Big‑M。
        - 对于 body<=0：需要body的上界U；M=max(U,0)
        - 对于 body>=0：需要body的下界L；M=max(-L,0)
        仅在线性情形下返回估计；否则返回None。
        """
        try:
            import sympy as sp
            body = sp.expand(body_sym_expr)
            # 仅支持所有变量二阶导为0
            for s in self.parse_result.all_symbols:
                if body.diff(s, 2) != 0:
                    return None
            # 抽取线性系数
            symbols = list(self.parse_result.sorted_symbols)
            poly = sp.Poly(body, *symbols)
            coeffs = {s: 0.0 for s in symbols}
            const = 0.0
            for monom, c in poly.terms():
                deg_sum = sum(monom)
                cval = float(c)
                if deg_sum == 0:
                    const += cval
                elif deg_sum == 1:
                    # 找到一次项的符号
                    for idx, powv in enumerate(monom):
                        if powv == 1:
                            coeffs[symbols[idx]] += cval
                            break
                else:
                    return None
            bm = self.parse_result.bound_manager
            xl = np.array(bm.lower_bounds, dtype=float)
            xu = np.array(bm.upper_bounds, dtype=float)
            U = const
            L = const
            for i, s in enumerate(symbols):
                a = float(coeffs.get(s, 0.0))
                lo = float(xl[i]) if np.isfinite(xl[i]) else -np.inf
                hi = float(xu[i]) if np.isfinite(xu[i]) else np.inf
                if not np.isfinite(lo) or not np.isfinite(hi):
                    return None
                # 上界：系数>=0 取hi，否则取lo
                U += a * (hi if a >= 0 else lo)
                # 下界：系数>=0 取lo，否则取hi
                L += a * (lo if a >= 0 else hi)
            if sense == 'le':
                return max(U, 0.0)
            elif sense == 'ge':
                return max(-L, 0.0)
            elif sense == 'eq':
                # 等式用左右两侧的松弛
                return max(max(U, 0.0), max(-L, 0.0))
            return None
        except Exception:
            return None
