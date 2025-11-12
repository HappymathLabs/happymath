import os
import sys
import sympy as sp
import pyomo.environ as pyo

from happymath.Opt.adapters.pyomo_dae_adapter import PyomoDAEAdapter
from happymath.Opt.functional.config import ODEIVPConfig, DomainConfig, ControlParamConfig
from happymath.Opt.opt_expr.processor import ExpressionProcessor


def build_ivp_with_control():
    t = sp.symbols('t', real=True)
    x = sp.Function('x')
    u = sp.Function('u')
    ode = [sp.Eq(sp.diff(x(t), t, 1), -x(t) + u(t))]
    coeffs = sp.symbols('c0:3', real=True)
    cfg = ODEIVPConfig(
        ode=ode,
        domain=DomainConfig(var=t, t0=0.0, t1=1.0, grid_n=31),
        ivp_conds={x(0): 1.0},
        control=ControlParamConfig(kind='piecewise_constant', func=u, coeff_symbols=list(coeffs), segments=3, bounds=(-1.0, 1.0)),
        objective_meta={0: {"aggregation": "integral", "expr": x(t)**2}},
        extra_symbols=list(coeffs),
        bounds={s: (-1.0, 1.0) for s in coeffs},
    )
    ep = ExpressionProcessor()
    pr = ep.process({"min": x(t)**2}, None, functional_config=cfg)
    return pr, cfg


def test_pyomo_dae_control_parameterization_builds():
    pr, cfg = build_ivp_with_control()
    adapter = PyomoDAEAdapter(pr, cfg)
    m = adapter.convert()
    assert isinstance(m, pyo.ConcreteModel)
    # Control segment variables and per-time-point value variables exist
    assert any(name.endswith('_seg') for name, obj in m.component_map(pyo.Var, active=True).items())
    assert any(name.endswith('_val') for name, obj in m.component_map(pyo.Var, active=True).items())
    # Binding constraints exist
    assert any(name.endswith('_link') for name, obj in m.component_map(pyo.Constraint, active=True).items())

