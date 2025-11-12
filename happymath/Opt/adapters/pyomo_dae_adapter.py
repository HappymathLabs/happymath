"""
Pyomo DAE 适配器（M3）

功能：
- 将 ODE/IVP 的功能型配置（ODEIVPConfig）构造成 Pyomo.DAE 模型；
- 支持：状态变量/导数变量、初值、ODE 约束、积分状态法（Integral 目标/约束）、离散化（collocation）。

限制：
- 一期不支持控制参数化（u(t)）直接建模；如存在控制，请使用 Pymoo 路径或将控制视作已知函数（后续扩展）。
"""

from __future__ import annotations

from typing import Any, Dict, List
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
import sympy as sp
from sympy.core.function import AppliedUndef

from ..functional.config import ODEIVPConfig


class PyomoDAEAdapter:
    """将 ODEIVPConfig 构造成 Pyomo.DAE 模型"""

    def __init__(self, parse_result, functional_cfg: ODEIVPConfig, epsilon: float = 1e-6,
                 nfe: int | None = None, ncp: int | None = None):
        self.parse_result = parse_result
        self.cfg: ODEIVPConfig = functional_cfg
        self.epsilon = float(epsilon)
        self.nfe = int(nfe) if nfe is not None else max(10, (int(getattr(functional_cfg.domain, 'grid_n', 101)) // 10))
        self.ncp = int(ncp) if ncp is not None else 3

        # 状态符号：从 ODE 等式左侧（Derivative(y(t), t)）提取 y(t)
        self._state_funcs: List[sp.Function] = []  # 形如 x(t)
        for eq in self.cfg.ode:
            if isinstance(eq.lhs, sp.Derivative):
                base = eq.lhs.args[0]
                if isinstance(base, AppliedUndef):
                    if base not in self._state_funcs:
                        self._state_funcs.append(base)

    # === 公共入口 ===
    def convert(self) -> pyo.ConcreteModel:
        m = pyo.ConcreteModel(name='DAE_ODE_IVP')
        t0 = float(self.cfg.domain.t0)
        t1 = float(self.cfg.domain.t1)
        m.t = ContinuousSet(bounds=(t0, t1))

        # 状态与导数
        m.state_vars: Dict[str, pyo.Var] = {}
        m.deriv_vars: Dict[str, DerivativeVar] = {}
        for f in self._state_funcs:
            name = str(f.func)
            v = pyo.Var(m.t)
            setattr(m, name, v)
            m.state_vars[name] = v
            dv = DerivativeVar(v, wrt=m.t)
            setattr(m, f'd{name}_dt', dv)
            m.deriv_vars[name] = dv

        # 控制参数化（若有）
        self._attach_control_parameterization(m)

        # 参数变量（可选）：来自 cfg.param_symbols
        self._param_var_map: Dict[Any, pyo.Var] = {}
        if getattr(self.cfg, 'param_symbols', None):
            for s in self.cfg.param_symbols:
                lb, ub = None, None
                try:
                    b = (self.cfg.param_bounds or {}).get(s)
                    if b is not None:
                        lb, ub = float(b[0]), float(b[1])
                except Exception:
                    pass
                par = pyo.Var(bounds=(lb, ub) if lb is not None and ub is not None else None)
                pname = f"p_{str(s)}"
                setattr(m, pname, par)
                self._param_var_map[s] = par

        # ODE 约束（逐方程索引化）
        eqs = [eq for eq in self.cfg.ode if isinstance(eq.lhs, sp.Derivative)]
        m.ode_i = pyo.RangeSet(0, len(eqs) - 1) if eqs else pyo.RangeSet(0, -1)

        def _ode_rule(mdl, i, tt):
            eq = eqs[i]
            base = eq.lhs.args[0]
            base_name = str(base.func)
            lhs = m.deriv_vars[base_name][tt]
            rhs = self._sympy_to_pyomo_expr(eq.rhs, mdl, tt)
            return lhs == rhs

        m.ode = pyo.Constraint(m.ode_i, m.t, rule=_ode_rule)

        # 先不离散化；待构造完目标与所有微分约束后统一离散化
        # 初值约束将在离散化之后添加（使用 m.t.first()）
        m.ic_list = pyo.ConstraintList()

        # 目标：Integral → 积分状态法；终端目标 final_state → 直接取末端
        # 仅处理单目标（Pyomo 不支持多目标）
        # 优先读取 cfg.objective_meta[0]
        meta = (self.cfg.objective_meta or {}).get(0, {})
        agg = meta.get('aggregation')
        has_integral_obj = False
        if agg == 'integral':
            expr_meta = meta.get('expr')  # 被积函数（不可含未知函数 x(t)，如含则在 _sympy_to_pyomo_expr 中处理）
            # 创建积分状态 I(t)
            m.I = pyo.Var(m.t)
            m.dIdt = DerivativeVar(m.I, wrt=m.t)
            # dI/dt = integrand
            def dI_rule(mdl, tt):
                rhs = self._sympy_to_pyomo_expr(expr_meta, mdl, tt)
                return mdl.dIdt[tt] == rhs
            m.integral_dyn = pyo.Constraint(m.t, rule=dI_rule)
            has_integral_obj = True
            # 目标：min I(t1) 或窗口 I(t_w1)-I(t_w0)
            w = (self.cfg.objective_meta or {}).get(0, {}).get('window') if isinstance((self.cfg.objective_meta or {}).get(0, {}), dict) else None
            if w is None:
                m.obj = pyo.Objective(expr=m.I[m.t.last()], sense=pyo.minimize)
            else:
                # 定位最近离散点
                t_points = list(m.t)
                t0w = float(w.get('t0', self.cfg.domain.t0))
                t1w = float(w.get('t1', self.cfg.domain.t1))
                t0_idx = min(range(len(t_points)), key=lambda k: abs(t_points[k] - t0w))
                t1_idx = min(range(len(t_points)), key=lambda k: abs(t_points[k] - t1w))
                m.obj = pyo.Objective(expr=m.I[t_points[t1_idx]] - m.I[t_points[t0_idx]], sense=pyo.minimize)
        elif agg == 'final_state':
            # expr_meta 为 x(t)；state_index 可选
            expr_meta = meta.get('expr')
            idx = int(meta.get('state_index', 0))
            # 简单实现：选第一个状态的末端为目标
            if self._state_funcs:
                vname = str(self._state_funcs[idx].func)
                m.obj = pyo.Objective(expr=getattr(m, vname)[m.t.last()], sense=pyo.minimize)
            else:
                # 回退：常数目标
                m.obj = pyo.Objective(expr=0.0)
        else:
            # 未提供可识别的聚合，回退为常数目标
            m.obj = pyo.Objective(expr=0.0)

        # 统一离散化（collocation）
        from pyomo.core import TransformationFactory
        TransformationFactory('dae.collocation').apply_to(m, nfe=self.nfe, ncp=self.ncp)

        # 离散化后设置初值（状态变量 + 积分状态）
        for k, v in (self.cfg.ivp_conds or {}).items():
            if isinstance(k, AppliedUndef):
                name = str(k.func)
                try:
                    val = float(v)
                except Exception:
                    continue
                m.ic_list.add(expr=getattr(m, name)[m.t.first()] == val)
        if has_integral_obj:
            m.ic_list.add(expr=m.I[m.t.first()] == 0.0)

        return m

    def _attach_control_parameterization(self, m: pyo.ConcreteModel) -> None:
        """在离散化后为控制 u(t) 创建分段常值变量与绑定约束。

        设计：
        - u_seg[k]: 第 k 段控制值（k=0..segments-1），具有统一边界；
        - u_val[t]: 时刻 t 的控制值；
        - 绑定约束：u_val[t] == u_seg[seg_index(t)]，seg_index 按 [t0,t1] 等分确定。
        - _sympy_to_pyomo_expr 遇到 control.func(t) 返回 u_val[t]。
        """
        ctrl = getattr(self.cfg, 'control', None)
        if ctrl is None or getattr(ctrl, 'func', None) is None:
            return
        name = str(ctrl.func)
        segments = int(getattr(ctrl, 'segments', 0) or 0)
        if segments <= 0:
            return
        # 段值变量
        lb_ub = getattr(ctrl, 'bounds', None)
        if lb_ub is not None:
            lb, ub = float(lb_ub[0]), float(lb_ub[1])
            u_seg = pyo.Var(range(segments), bounds=(lb, ub))
        else:
            u_seg = pyo.Var(range(segments))
        setattr(m, f'{name}_seg', u_seg)
        # 时序控制值
        u_val = pyo.Var(m.t)
        setattr(m, f'{name}_val', u_val)
        # 绑定约束：根据时间点归属段
        t0 = float(self.cfg.domain.t0)
        t1 = float(self.cfg.domain.t1)
        width = (t1 - t0) / segments
        def _seg_index(tt: float) -> int:
            if tt >= t1:
                return segments - 1
            k = int((tt - t0) / width)
            if k < 0:
                k = 0
            if k >= segments:
                k = segments - 1
            return k
        def _link_rule(mdl, tt):
            k = _seg_index(float(tt))
            return getattr(mdl, f'{name}_val')[tt] == getattr(mdl, f'{name}_seg')[k]
        setattr(m, f'{name}_link', pyo.Constraint(m.t, rule=_link_rule))

    # === SymPy → Pyomo 表达式转换（含时间索引）===
    def _sympy_to_pyomo_expr(self, expr: sp.Expr, mdl: pyo.ConcreteModel, tt) -> Any:
        # 常数
        if expr is None:
            return 0.0
        if isinstance(expr, (int, float)):
            return float(expr)
        if isinstance(expr, sp.Number):
            return float(expr)

        f = expr.func
        args = expr.args

        # 函数应用：x(t) → mdl.x[tt]
        if isinstance(expr, AppliedUndef):
            name = str(expr.func)
            # 若为控制函数，返回离散化后的 u_val[tt]
            ctrl = getattr(self.cfg, 'control', None)
            if ctrl is not None and getattr(ctrl, 'func', None) is not None and str(ctrl.func) == name:
                try:
                    return getattr(mdl, f'{name}_val')[tt]
                except Exception:
                    pass
            return getattr(mdl, name)[tt]
        # 导数：Derivative(x(t), t) → d{x}_dt[tt]
        if isinstance(expr, sp.Derivative):
            try:
                base = expr.args[0]
                if isinstance(base, AppliedUndef):
                    name = str(base.func)
                    return getattr(mdl, f'd{name}_dt')[tt]
            except Exception:
                pass

        # 基本代数
        from sympy import Add, Mul, Pow
        if f is Add:
            return sum(self._sympy_to_pyomo_expr(a, mdl, tt) for a in args)
        if f is Mul:
            prod = 1
            for a in args:
                prod = prod * self._sympy_to_pyomo_expr(a, mdl, tt)
            return prod
        if f is Pow:
            # 安全处理负指数：base**(-n) → 1.0 / (base_safe**n)
            base = self._sympy_to_pyomo_expr(args[0], mdl, tt)
            expn = args[1]
            if isinstance(expn, (int, float, sp.Number)):
                try:
                    e = float(expn)
                except Exception:
                    e = None
                if e is not None and e < 0:
                    # 对底数加入极小正数，避免 0 的负次方
                    base_safe = base + 1e-12
                    return 1.0 / (base_safe ** abs(e))
                else:
                    return base ** float(expn)
            else:
                return base ** self._sympy_to_pyomo_expr(expn, mdl, tt)

        # 常用函数
        from sympy import sin, cos, tan, exp, log, sqrt, Abs
        if f is sin:
            return pyo.sin(self._sympy_to_pyomo_expr(args[0], mdl, tt))
        if f is cos:
            return pyo.cos(self._sympy_to_pyomo_expr(args[0], mdl, tt))
        if f is tan:
            return pyo.tan(self._sympy_to_pyomo_expr(args[0], mdl, tt))
        if f is exp:
            return pyo.exp(self._sympy_to_pyomo_expr(args[0], mdl, tt))
        if f is log:
            return pyo.log(self._sympy_to_pyomo_expr(args[0], mdl, tt))
        if f is sqrt:
            inner = self._sympy_to_pyomo_expr(args[0], mdl, tt)
            return inner ** 0.5
        if f is Abs:
            inner = self._sympy_to_pyomo_expr(args[0], mdl, tt)
            # 绝对值在 NLP 中可通过平方根近似或转化；此处直接采用二范数近似（近似实现）
            return pyo.sqrt(inner ** 2)

        # 变量符号（常数/参数）：优先映射到参数变量，其次尝试直接浮点化
        try:
            if hasattr(self, '_param_var_map') and expr in self._param_var_map:
                return self._param_var_map[expr]
        except Exception:
            pass
        try:
            return float(expr)
        except Exception:
            pass

        # 未识别：回退 0
        return 0.0


__all__ = ["PyomoDAEAdapter"]
