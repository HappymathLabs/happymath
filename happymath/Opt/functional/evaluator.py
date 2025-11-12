"""
FUNCTIONAL 评估器（Pymoo / Pyomo 共用）

增强版说明（P1）：
- 支持通用轨迹积分 integrand(expr(t, x(t), x'(t), ... , params))，可选时间窗口；
- 支持导数组合算子：若 DiffEq 未一阶化输出，则在窗口内进行中心差分；
- 一次仿真多指标共享缓存（GLOBAL_SIM_CACHE），窗口签名纳入缓存键；
- 保持 FunctionalSpec 接口，对外兼容；

设计准则：中文注释；高内聚低耦合；不侵入 DiffEq，仅复用其标准化信息与数值求解。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Optional
import numpy as np
import sympy as sp
from sympy.core.function import AppliedUndef

from .spec import FunctionalSpec
import os, sys

# 兼容本地源码布局：确保 DiffEq 子包可被导入
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_DIFFEQ = os.path.join(_ROOT, 'DiffEq')
for _p in (_ROOT, _DIFFEQ):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from .config import ODEIVPConfig, ODEBVPConfig, ControlParamConfig, MetricSpec, IntegrandSpec, WindowSpec


def _piecewise_constant_expr(t: sp.Symbol, coeffs: List[sp.Symbol], t0: float, t1: float) -> sp.Expr:
    """构造分段常数 Piecewise 表达式 u(t)

    说明：
    - 将 [t0, t1] 均分为 len(coeffs) 段，每段取对应的系数值。
    - 返回的对象为 SymPy 表达式，便于后续替换到 ODE/目标 integrand 中。
    """
    n = len(coeffs)
    if n <= 0:
        raise ValueError("coeffs 不能为空")
    seg_edges = [t0 + (t1 - t0) * i / n for i in range(n + 1)]
    # 末段包含右端点
    pieces = []
    for k in range(n):
        left = seg_edges[k]
        right = seg_edges[k + 1]
        cond = sp.And(sp.Ge(t, left), sp.Le(t, right) if k == n - 1 else sp.Lt(t, right))
        pieces.append((coeffs[k], cond))
    return sp.Piecewise(*pieces)


def _substitute_control(ode_list: List[sp.Eq], control: ControlParamConfig, domain_t: sp.Symbol, t0: float, t1: float) -> Tuple[List[sp.Eq], Dict[str, Any]]:
    """将 ODE 中的控制函数 u(t) 用分段常数的符号表达式替换

    返回：
      - 新的 ODE 方程列表
      - 辅助信息字典（包含 'ctrl_expr' 便于目标评估时复用）
    """
    if control is None or control.func is None:
        return ode_list, {}

    coeffs = list(control.coeff_symbols or [])
    if len(coeffs) == 0:
        raise ValueError("控制参数化缺少 coeff_symbols")
    ctrl_expr = _piecewise_constant_expr(domain_t, coeffs, t0, t1)

    new_list = []
    for eq in ode_list:
        rhs = eq.rhs.subs({control.func: sp.Lambda(domain_t, ctrl_expr)}).doit() if isinstance(eq.rhs, sp.Integral) else eq.rhs
        # 对 u(t) 直接替换为分段常数表达式（不使用 Lambda，避免 lambdify 时引入 Python 回调）
        rhs = rhs.xreplace({control.func(domain_t): ctrl_expr})
        # 同时替换等式左边（一般为 Derivative(y(t), t, k)），通常无需替，但保持一致
        lhs = eq.lhs.xreplace({control.func(domain_t): ctrl_expr})
        new_list.append(sp.Eq(lhs, rhs))

    return new_list, {"ctrl_expr": ctrl_expr}


def _central_diff(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """对向量 y(t) 进行中心差分，端点使用一侧差分。"""
    n = len(y)
    if n != len(t):
        raise ValueError("y 与 t 长度不一致")
    if n < 2:
        return np.zeros_like(y)
    dy = np.zeros_like(y, dtype=float)
    for i in range(1, n - 1):
        dy[i] = (y[i + 1] - y[i - 1]) / (t[i + 1] - t[i - 1])
    dy[0] = (y[1] - y[0]) / (t[1] - t[0])
    dy[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
    return dy


def _mask_window(t: np.ndarray, w: Optional[WindowSpec]) -> np.ndarray:
    """根据窗口规格生成布尔掩码；None 表示全窗口。"""
    if w is None:
        return np.ones_like(t, dtype=bool)
    lo = min(float(w.t0), float(w.t1))
    hi = max(float(w.t0), float(w.t1))
    return (t >= lo) & (t <= hi)


def _build_metric_specs_from_meta(cfg: ODEIVPConfig) -> List[MetricSpec]:
    """从 objective_meta / constraint_meta 兼容生成指标规格。"""
    if cfg.metrics:
        return list(cfg.metrics)
    specs: List[MetricSpec] = []
    # 目标
    for key, meta in (cfg.objective_meta or {}).items():
        agg = str(meta.get("aggregation", "")).lower()
        if agg == "integral":
            expr = meta.get("expr")
            specs.append(MetricSpec(
                id=f"obj:{key}",
                kind='integral',
                integrand=IntegrandSpec(id=f"obj:{key}:integrand", expr=expr, window=None),
                agg='trapz',
            ))
        elif agg in {"final_state", "terminal"}:
            specs.append(MetricSpec(
                id=f"obj:{key}",
                kind='terminal',
                state_index=int(meta.get("state_index", 0)),
                agg='terminal',
            ))
        elif agg in {"l2_norm", "l1_norm", "max_abs", "mean_abs"}:
            specs.append(MetricSpec(
                id=f"obj:{key}",
                kind='path',
                state_index=int(meta.get("state_index", 0)),
                agg=agg,
            ))
    # 约束
    for key, meta in (cfg.constraint_meta or {}).items():
        agg = str(meta.get("aggregation", "")).lower()
        if agg == "integral":
            expr = meta.get("expr")
            specs.append(MetricSpec(
                id=f"con:{key}",
                kind='integral',
                integrand=IntegrandSpec(id=f"con:{key}:integrand", expr=expr, window=None),
                agg='trapz',
            ))
        elif agg in {"final_state", "terminal"}:
            specs.append(MetricSpec(
                id=f"con:{key}",
                kind='terminal',
                state_index=int(meta.get("state_index", 0)),
                agg='terminal',
            ))
        elif agg in {"l2_norm", "l1_norm", "max_abs", "mean_abs"}:
            specs.append(MetricSpec(
                id=f"con:{key}",
                kind='path',
                state_index=int(meta.get("state_index", 0)),
                agg=agg,
            ))
    return specs


class TrajectoryEvaluator:
    """轨迹评估器

    职责：
    - 编译系统（控制替换、标准化信息获取）；
    - 执行一次仿真并缓存；
    - 在一次仿真结果上计算多个指标（含窗口积分、终端/路径聚合、导数组合等）。
    """

    def __init__(self, cfg: ODEIVPConfig):
        self.cfg = cfg
        self._compiled = False
        self._ode_param: List[sp.Eq] = []
        self._aux: Dict[str, Any] = {}
        self._state_funcs: List[AppliedUndef] = []
        self._grid: Optional[np.ndarray] = None

    def compile(self) -> None:
        """预编译系统：完成控制替换并提取状态函数列表。"""
        t = self.cfg.domain.var
        t0, t1 = float(self.cfg.domain.t0), float(self.cfg.domain.t1)
        self._ode_param, self._aux = _substitute_control(self.cfg.ode, self.cfg.control, t, t0, t1)
        funcs = []
        for eq in self._ode_param:
            if isinstance(eq.lhs, sp.Derivative) and isinstance(eq.lhs.args[0], AppliedUndef):
                base = eq.lhs.args[0]
                if base not in funcs:
                    funcs.append(base)
        self._state_funcs = funcs
        self._compiled = True

    def _eval_to_float(self, val: Any, opt_vars: Dict[str, float]) -> float:
        """将符号/数值安全转换为 float，并用 opt_vars 进行替换。"""
        try:
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, sp.Basic):
                syms = list(val.free_symbols)
                if not syms:
                    return float(val.evalf())
                subs = {s: opt_vars.get(str(s)) for s in syms}
                return float(val.subs(subs).evalf())
            return float(opt_vars.get(str(val), 0.0))
        except Exception:
            return float(sp.N(val)) if isinstance(val, sp.Basic) else float(val)

    def simulate(self, opt_vars: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """执行一次仿真，返回 (t, Y)。缓存键包含 cfg id + 决策变量快照 + N。"""
        if not self._compiled:
            self.compile()
        t = self.cfg.domain.var
        t0, t1, N = float(self.cfg.domain.t0), float(self.cfg.domain.t1), int(self.cfg.domain.grid_n or 101)
        grid = np.linspace(t0, t1, N)
        self._grid = grid
        # 构建缓存键
        keys = []
        if self.cfg.control and self.cfg.control.coeff_symbols:
            keys.extend([str(s) for s in self.cfg.control.coeff_symbols])
        if getattr(self.cfg, 'param_symbols', None):
            keys.extend([str(s) for s in self.cfg.param_symbols])
        if getattr(self.cfg, 'extra_symbols', None):
            keys.extend([str(s) for s in self.cfg.extra_symbols])
        snap = tuple(sorted((k, float(opt_vars.get(k, 0.0))) for k in keys))
        cache_key = (id(self.cfg), snap, N)
        if cache_key in GLOBAL_SIM_CACHE:
            Y = GLOBAL_SIM_CACHE[cache_key]
            return grid, Y

        # 若存在控制参数化（通常引入 Piecewise），直接走 SciPy 轻量路径；否则优先 DiffEq
        use_scipy_only = bool(self.cfg.control and self.cfg.control.func is not None)
        try:
            if use_scipy_only:
                raise RuntimeError("skip-DiffEq-for-piecewise-control")
            from DiffEq.ODE.ODEModule import ODEModule
            ics = {}
            for k, v in (self.cfg.ivp_conds or {}).items():
                ics[k] = self._eval_to_float(v, opt_vars)
            consts = {}
            for s, v in (self.cfg.constants or {}).items():
                consts[s] = self._eval_to_float(v, opt_vars)
            for s in getattr(self.cfg, 'param_symbols', []) or []:
                if str(s) in opt_vars:
                    consts[s] = float(opt_vars[str(s)])
            ode_mod = ODEModule(self._ode_param)
            sol = ode_mod.num_solve("IVP", ics, domain=grid, const_cond=consts, solve_method="RK45", tol=1e-6)
            Y = np.asarray(sol, dtype=float)
        except Exception:
            # SciPy 轻量回退：构造向量场 f(t, y)
            from scipy.integrate import solve_ivp
            # 状态函数占位符
            state_syms = [sp.Symbol(f"__s{i}") for i in range(len(self._state_funcs) or 1)]
            rhs_funcs: List[Callable] = []
            # 构造每个方程的 RHS 函数
            for idx, eq in enumerate(self._ode_param):
                expr = eq.rhs
                # 替换状态函数为占位符
                subs_map = {}
                for j, f in enumerate(self._state_funcs):
                    subs_map[f] = state_syms[j]
                expr_use = expr.xreplace(subs_map)
                # 将常数与参数以数值替换到表达式中，减少 lambdify 入参
                numeric_subs = {}
                for s, v in (self.cfg.constants or {}).items():
                    numeric_subs[s] = self._eval_to_float(v, opt_vars)
                for s in getattr(self.cfg, 'param_symbols', []) or []:
                    if str(s) in opt_vars:
                        numeric_subs[s] = float(opt_vars[str(s)])
                # 控制系数数值替换（避免 lambdify 外再传参）
                if self.cfg.control and self.cfg.control.coeff_symbols:
                    for s in self.cfg.control.coeff_symbols:
                        if str(s) in opt_vars:
                            numeric_subs[s] = float(opt_vars[str(s)])
                if numeric_subs:
                    expr_use = expr_use.subs(numeric_subs)
                lam = sp.lambdify([self.cfg.domain.var] + state_syms, expr_use, "numpy")
                rhs_funcs.append(lam)

            # 初值向量 y0 按状态函数顺序提取
            y0: List[float] = []
            ics_dict = self.cfg.ivp_conds or {}
            t0_val = float(self.cfg.domain.t0)
            for f in (self._state_funcs or []):
                # 支持 x(t0) 或 x(0)
                cand = [ics_dict.get(f.subs({self.cfg.domain.var: t0_val}), None), ics_dict.get(f.subs({self.cfg.domain.var: 0}), None)]
                ival = next((c for c in cand if c is not None), 0.0)
                y0.append(float(self._eval_to_float(ival, opt_vars)))
            if not y0:
                y0 = [0.0]

            def vf(tval, yvec):
                # 保持与 lambdify 入参一致
                args = [tval] + list(yvec)
                return [float(fn(*args)) for fn in rhs_funcs]

            sol = solve_ivp(lambda tt, yy: vf(tt, yy), (t0, t1), np.asarray(y0, dtype=float), t_eval=grid, rtol=1e-6, atol=1e-9, method='RK45')
            if not sol.success:
                raise RuntimeError(f"solve_ivp 失败: {sol.message}")
            Y = np.asarray(sol.y.T, dtype=float)
        GLOBAL_SIM_CACHE[cache_key] = Y
        return grid, Y

    def evaluate_all(self, opt_vars: Dict[str, float], metrics: Optional[List[MetricSpec]] = None) -> Dict[str, float]:
        """一次仿真，评估多个指标。返回：id -> 值。"""
        if metrics is None or len(metrics) == 0:
            metrics = _build_metric_specs_from_meta(self.cfg)
        t, Y = self.simulate(opt_vars)
        res: Dict[str, float] = {}
        # 预计算导数矩阵
        Ydot = np.zeros_like(Y)
        for j in range(Y.shape[1]):
            Ydot[:, j] = _central_diff(Y[:, j], t)
        # 控制/参数替换
        ctrl_expr = self._aux.get("ctrl_expr") if (self.cfg.control and self._aux) else None
        t_sym = self.cfg.domain.var
        # 基函数名到列索引的映射（若不可识别则默认第0列）
        func_to_col: Dict[str, int] = {}
        for idx, f in enumerate(self._state_funcs):
            func_to_col[str(f.func)] = idx

        for m in metrics:
            if m.kind == 'integral' and m.integrand is not None:
                wmask = _mask_window(t, m.integrand.window)
                expr = m.integrand.expr
                if expr is None:
                    res[m.id] = 0.0
                    continue
                # 若 integrand 不含状态/导数，仅含 u(t)、t、参数：优先走专用路径（避免 Piecewise → object 数组问题）
                atoms_funcs = list(expr.atoms(AppliedUndef))
                state_atoms = [f for f in atoms_funcs if self.cfg.control and f.func != self.cfg.control.func]
                deriv_atoms = list(expr.atoms(sp.Derivative))
                param_syms = list(getattr(self.cfg, 'param_symbols', []) or [])
                if len(state_atoms) == 0 and len(deriv_atoms) == 0 and self.cfg.control and self.cfg.control.func is not None:
                    # 计算 u(t) 数组
                    u_expr = ctrl_expr if ctrl_expr is not None else self.cfg.control.func(t_sym)
                    # 将控制系数替换为数值
                    if self.cfg.control and self.cfg.control.coeff_symbols:
                        for s in self.cfg.control.coeff_symbols:
                            val = float(opt_vars.get(str(s), 0.0))
                            u_expr = u_expr.subs({s: val})
                    # 常数替换（若 integrand 中包含常数符号）
                    for s, v in (getattr(self.cfg, 'constants', {}) or {}).items():
                        try:
                            u_expr = u_expr.xreplace({s: float(v)})
                        except Exception:
                            pass
                    u_np = sp.lambdify([t_sym], u_expr, "numpy")(t)
                    # 将 integrand 中的 u(t) 替换为 U 占位变量
                    U = sp.Symbol("__U")
                    expr_u = expr.xreplace({self.cfg.control.func(t_sym): U})
                    # 常数替换
                    for s, v in (getattr(self.cfg, 'constants', {}) or {}).items():
                        try:
                            expr_u = expr_u.xreplace({s: float(v)})
                        except Exception:
                            pass
                    f_np = sp.lambdify([t_sym, U] + param_syms, expr_u, "numpy")
                    param_vals = [float(opt_vars.get(str(ps), 0.0)) for ps in param_syms]
                    vals = f_np(t, u_np, *param_vals)
                    vals = np.asarray(vals, dtype=float)
                else:
                    # 通用路径：替换状态/导数为占位符；控制采用数值化占位符 U，避免 Piecewise → object 数组
                    f_syms: List[sp.Symbol] = []
                    fd_syms: List[sp.Symbol] = []
                    subs_map = {}
                    for node in sorted(expr.atoms(AppliedUndef), key=lambda x: str(x.func)):
                        base_name = str(node.func)
                        # 控制函数不按状态处理，统一用 U 注入
                        if self.cfg.control and self.cfg.control.func is not None and base_name == str(self.cfg.control.func):
                            continue
                        s = sp.Symbol(f"__state__{base_name}")
                        subs_map[node] = s
                        f_syms.append(s)
                    for dnode in expr.atoms(sp.Derivative):
                        try:
                            if isinstance(dnode.args[0], AppliedUndef):
                                base_name = str(dnode.args[0].func)
                                if self.cfg.control and self.cfg.control.func is not None and base_name == str(self.cfg.control.func):
                                    continue
                                d = sp.Symbol(f"__dstate__{base_name}")
                                subs_map[dnode] = d
                                fd_syms.append(d)
                        except Exception:
                            continue
                    expr_use = expr.xreplace(subs_map)
                    # 常数替换
                    for s, v in (getattr(self.cfg, 'constants', {}) or {}).items():
                        try:
                            expr_use = expr_use.xreplace({s: float(v)})
                        except Exception:
                            pass
                    U = None
                    if self.cfg.control and self.cfg.control.func is not None and expr.has(self.cfg.control.func(t_sym)):
                        U = sp.Symbol("__U")
                        expr_use = expr_use.xreplace({self.cfg.control.func(t_sym): U})
                    arg_syms: List[Any] = [t_sym]
                    arg_syms.extend(f_syms)
                    arg_syms.extend(fd_syms)
                    arg_syms.extend(param_syms)
                    if U is not None:
                        arg_syms.append(U)
                    f_np = sp.lambdify(arg_syms, expr_use, "numpy")
                    # 组织数值入参
                    f_vals: List[Any] = []
                    for s in f_syms:
                        name = s.name.replace("__state__", "")
                        col = func_to_col.get(name, 0)
                        f_vals.append(Y[:, col])
                    for ds in fd_syms:
                        name = ds.name.replace("__dstate__", "")
                        col = func_to_col.get(name, 0)
                        f_vals.append(Ydot[:, col])
                    for ps in param_syms:
                        f_vals.append(float(opt_vars.get(str(ps), 0.0)))
                    if U is not None:
                        u_expr = ctrl_expr if ctrl_expr is not None else self.cfg.control.func(t_sym)
                        if self.cfg.control and self.cfg.control.coeff_symbols:
                            for s in self.cfg.control.coeff_symbols:
                                val = float(opt_vars.get(str(s), 0.0))
                                u_expr = u_expr.subs({s: val})
                        u_np = sp.lambdify([t_sym], u_expr, "numpy")(t)
                        f_vals.append(u_np)
                    vals = f_np(t, *f_vals)
                    vals = np.asarray(vals, dtype=float)
                vals_win = vals[_mask_window(t, m.integrand.window)]
                t_win = t[_mask_window(t, m.integrand.window)]
                if m.agg == 'simpson' and len(t_win) >= 3:
                    try:
                        from scipy.integrate import simps
                        res[m.id] = float(simps(vals_win, t_win))
                        continue
                    except Exception:
                        pass
                res[m.id] = float(np.trapz(vals_win, t_win))
            elif m.kind == 'terminal':
                idx = int(m.state_index or 0)
                res[m.id] = float(Y[-1, idx])
            elif m.kind == 'path':
                idx = int(m.state_index or 0)
                vec = Y[:, idx]
                agg = (m.agg or 'l2_norm').lower()
                if agg == 'l2_norm':
                    res[m.id] = float(np.sqrt(np.sum(vec ** 2)))
                elif agg == 'l1_norm':
                    res[m.id] = float(np.sum(np.abs(vec)))
                elif agg == 'max_abs':
                    res[m.id] = float(np.max(np.abs(vec)))
                elif agg == 'mean_abs':
                    res[m.id] = float(np.mean(np.abs(vec)))
                else:
                    res[m.id] = float(np.sqrt(np.sum(vec ** 2)))
            else:
                res[m.id] = 0.0
        return res


def build_ode_ivp_evaluator(cfg: ODEIVPConfig, objective_key: Any, meta_override: Dict[str, Any] | None = None) -> FunctionalSpec:
    """构造兼容式 evaluator：内部使用 TrajectoryEvaluator 单指标取值。"""
    t = cfg.domain.var
    t0, t1, N = cfg.domain.t0, cfg.domain.t1, int(cfg.domain.grid_n or 101)
    traj = TrajectoryEvaluator(cfg)

    # 组装单指标 MetricSpec
    if meta_override is not None:
        agg = str(meta_override.get('aggregation', '')).lower()
        if agg == 'integral':
            expr = meta_override.get('expr')
            metrics = [MetricSpec(id=f"obj:{objective_key}", kind='integral', integrand=IntegrandSpec(id=f"obj:{objective_key}:integrand", expr=expr))]
        elif agg in {'final_state', 'terminal'}:
            metrics = [MetricSpec(id=f"obj:{objective_key}", kind='terminal', state_index=int(meta_override.get('state_index', 0))) ]
        else:
            metrics = [MetricSpec(id=f"obj:{objective_key}", kind='path', state_index=int(meta_override.get('state_index', 0)), agg=agg or 'l2_norm')]
    else:
        metrics = _build_metric_specs_from_meta(cfg)
        metrics = [m for m in metrics if m.id == f"obj:{objective_key}"] or metrics[:1]

    def evaluator(opt_vars: Dict[str, float]) -> float:
        all_vals = traj.evaluate_all(opt_vars, metrics)
        return float(all_vals.get(metrics[0].id, 0.0))

    spec = FunctionalSpec(
        evaluator=evaluator,
        metadata={
            "kind": "ode_ivp",
            "domain": {"var": t, "t0": t0, "t1": t1, "N": N},
            "aggregation": getattr(metrics[0], 'agg', None),
        },
    )
    return spec


__all__ = ["build_ode_ivp_evaluator", "TrajectoryEvaluator"]

# 简单的全局仿真缓存：避免同一 x 重复仿真
GLOBAL_SIM_CACHE: Dict[Any, Any] = {}


# === P3：BVP 评估器（简化版）===

class TrajectoryEvaluatorBVP:
    """BVP 轨迹评估器（简化）：

    - 支持 x'(t)=f(...) 的边值问题，条件限制为 x(t0)=v、x(t1)=v（不包含导数在评估条件中）。
    - 数值仿真依赖 DiffEq 的 BVP 适配器构造；若失败则直接抛错（不做回退）。
    - 指标评估与 IVP 类似：integral/terminal/path 三类（窗口对 BVP 保留为全域）。
    """

    def __init__(self, cfg: ODEBVPConfig):
        self.cfg = cfg
        self._compiled = False
        self._ode = []
        self._state_funcs: List[AppliedUndef] = []

    def compile(self) -> None:
        # BVP 不做控制替换（一期）；直接提取状态函数
        self._ode = list(self.cfg.ode)
        funcs = []
        for eq in self._ode:
            if isinstance(eq.lhs, sp.Derivative) and isinstance(eq.lhs.args[0], AppliedUndef):
                base = eq.lhs.args[0]
                if base not in funcs:
                    funcs.append(base)
        self._state_funcs = funcs
        self._compiled = True

    def simulate(self, opt_vars: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        if not self._compiled:
            self.compile()
        t0, t1 = float(self.cfg.domain.t0), float(self.cfg.domain.t1)
        N = 1 + int((self.cfg.domain.grid_n or 101) // 1)
        grid = np.linspace(t0, t1, N)

        # 准备条件与常数
        cond = dict(self.cfg.bvp_conds or {})
        consts = {}
        for s, v in (self.cfg.constants or {}).items():
            consts[s] = float(v)
        for s in getattr(self.cfg, 'param_symbols', []) or []:
            if str(s) in opt_vars:
                consts[s] = float(opt_vars[str(s)])

        # 尝试 DiffEq BVP；若失败，提供“零 RHS + 零边界”的解析回退
        try:
            from DiffEq.ODE.ODEModule import ODEModule
            ode_mod = ODEModule(self._ode)
            # BVP 校验需要提供 bc 函数；实际构造由适配器完成，此处提供占位满足验证
            import numpy as _np
            def _bc_dummy(ya, yb):
                try:
                    return _np.zeros_like(ya)
                except Exception:
                    return 0.0
            # 初始猜测：linear
            sol = ode_mod.num_solve("BVP", cond, domain=grid, const_cond=consts, bc=_bc_dummy, init_guess="linear", solve_method="RK45", tol=1e-4)
            Y = np.asarray(sol, dtype=float)
            return grid, Y
        except Exception:
            # 解析回退：若 RHS 全为 0 且边界值全为 0，则解恒为 0
            all_zero_rhs = all((eq.rhs.simplify() == 0) for eq in self._ode)
            all_zero_bc = True
            for k, v in (self.cfg.bvp_conds or {}).items():
                try:
                    if float(v) != 0.0:
                        all_zero_bc = False
                        break
                except Exception:
                    all_zero_bc = False
                    break
            if all_zero_rhs and all_zero_bc:
                n_state = max(1, len(self._state_funcs) or 1)
                Y = np.zeros((len(grid), n_state), dtype=float)
                return grid, Y
            raise

    def evaluate_all(self, opt_vars: Dict[str, float], metrics: Optional[List[MetricSpec]] = None) -> Dict[str, float]:
        if metrics is None or len(metrics) == 0:
            metrics = []
        t, Y = self.simulate(opt_vars)
        res: Dict[str, float] = {}
        # 路径与终端与 IVP 相同（BVP 暂不使用窗口）
        for m in metrics:
            if m.kind == 'terminal':
                idx = int(m.state_index or 0)
                res[m.id] = float(Y[-1, idx])
            elif m.kind == 'path':
                idx = int(m.state_index or 0)
                vec = Y[:, idx]
                agg = (m.agg or 'l2_norm').lower()
                if agg == 'l2_norm':
                    res[m.id] = float(np.sqrt(np.sum(vec ** 2)))
                elif agg == 'l1_norm':
                    res[m.id] = float(np.sum(np.abs(vec)))
                elif agg == 'max_abs':
                    res[m.id] = float(np.max(np.abs(vec)))
                elif agg == 'mean_abs':
                    res[m.id] = float(np.mean(np.abs(vec)))
                else:
                    res[m.id] = float(np.sqrt(np.sum(vec ** 2)))
            elif m.kind == 'integral' and m.integrand is not None:
                # 简化：不含状态导数的被积函数；含状态时按 IVP 同法可拓展
                expr = m.integrand.expr
                t_sym = self.cfg.domain.var
                # 构造状态映射（默认第一列）
                idx = int(m.state_index or 0)
                # 仅支持不含未知函数的表达式（如常数/参数函数），否则报错
                if expr.has(AppliedUndef):
                    raise ValueError("BVP integral 暂不支持含未知函数的 integrand。")
                f = sp.lambdify([t_sym] + list(getattr(self.cfg, 'param_symbols', []) or []), expr, "numpy")
                param_vals = [float(opt_vars.get(str(ps), 0.0)) for ps in (self.cfg.param_symbols or [])]
                vals = f(t, *param_vals)
                res[m.id] = float(np.trapz(vals, t))
            else:
                res[m.id] = 0.0
        return res
