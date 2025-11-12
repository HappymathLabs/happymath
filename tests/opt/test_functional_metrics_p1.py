import os
import sys
import numpy as np
import sympy as sp

from happymath.Opt.OptModule import OptModule
from happymath.Opt.functional.config import ODEIVPConfig, DomainConfig, ControlParamConfig, MetricSpec, IntegrandSpec, WindowSpec


def _build_cfg_with_metrics():
    # x'(t) = -x(t) + u(t), x(0)=1
    t = sp.symbols('t', real=True)
    x = sp.Function('x')
    u = sp.Function('u')

    ode = [sp.Eq(sp.diff(x(t), t, 1), -x(t) + u(t))]
    coeffs = sp.symbols('c0:4', real=True)

    # 自定义 metrics：
    #  - J1 = \int_{0.5}^{1.0} u^2 dt
    #  - J2 = 末端 x(1)
    #  - J3 = 路径 l2_norm(x)
    metrics = [
        MetricSpec(
            id='obj:J1',
            kind='integral',
            integrand=IntegrandSpec(id='J1:int', expr=u(t)**2, window=WindowSpec(0.5, 1.0)),
            agg='trapz',
        ),
        MetricSpec(id='obj:J2', kind='terminal', state_index=0, agg='terminal'),
        MetricSpec(id='obj:J3', kind='path', state_index=0, agg='l2_norm'),
    ]

    cfg = ODEIVPConfig(
        ode=ode,
        domain=DomainConfig(var=t, t0=0.0, t1=1.0, grid_n=201),
        ivp_conds={x(0): 1.0},
        control=ControlParamConfig(kind='piecewise_constant', func=u, coeff_symbols=list(coeffs), segments=4, bounds=(-2.0, 2.0)),
        extra_symbols=list(coeffs),
        bounds={s: (-2.0, 2.0) for s in coeffs},
        metrics=metrics,
    )
    return cfg, coeffs


def test_metrics_window_integral_and_terminal():
    cfg, coeffs = _build_cfg_with_metrics()
    # 目标占位，走 Pymoo evaluator
    obj = {"min": sp.Integer(0)}

    cons = []
    for c in coeffs:
        cons.append(c <= 2.0)
        cons.append(c >= -2.0)

    opt = OptModule(obj, cons, mode='pymoo', default_search_range=2.0, functional_config=cfg)
    # 缩短单测预算
    try:
        opt.pymoo_solver._budget_override = 80
    except Exception:
        pass
    res = opt.solve(solver='GA', use_auto_solvers=False, max_solvers=1)
    assert res is not None
    succ = any(r.get('success') for r in res.raw_all_solutions)
    assert succ


def test_evaluator_derivative_combo_delta_v():
    # 双状态示例：x_f, x_s；构造 Δv = x_s' - x_f' 并在窗口积分 Δv^2
    t = sp.symbols('t', real=True)
    x_f = sp.Function('x_f')
    x_s = sp.Function('x_s')
    u = sp.Function('u')

    ode = [
        sp.Eq(sp.diff(x_f(t), t, 1), -x_f(t) + u(t)),
        sp.Eq(sp.diff(x_s(t), t, 1), -0.5*x_s(t) + 0.5*u(t)),
    ]
    coeffs = sp.symbols('c0:3', real=True)

    dv = sp.diff(x_s(t), t) - sp.diff(x_f(t), t)
    metrics = [
        MetricSpec(
            id='obj:Jdv',
            kind='integral',
            integrand=IntegrandSpec(id='Jdv:int', expr=dv**2, window=WindowSpec(0.2, 0.8)),
            agg='trapz',
        ),
    ]

    cfg = ODEIVPConfig(
        ode=ode,
        domain=DomainConfig(var=t, t0=0.0, t1=1.0, grid_n=301),
        ivp_conds={x_f(0): 0.0, x_s(0): 0.0},
        control=ControlParamConfig(kind='piecewise_constant', func=u, coeff_symbols=list(coeffs), segments=3, bounds=(-1.0, 1.0)),
        extra_symbols=list(coeffs),
        bounds={s: (-1.0, 1.0) for s in coeffs},
        metrics=metrics,
    )
    obj = {"min": sp.Integer(0)}
    cons = []
    for c in coeffs:
        cons.append(c <= 1.0)
        cons.append(c >= -1.0)

    opt = OptModule(obj, cons, mode='pymoo', default_search_range=1.0, functional_config=cfg)
    try:
        opt.pymoo_solver._budget_override = 60
    except Exception:
        pass
    res = opt.solve(solver='GA', use_auto_solvers=False, max_solvers=1)
    assert res is not None
    succ = any(r.get('success') for r in res.raw_all_solutions)
    assert succ

