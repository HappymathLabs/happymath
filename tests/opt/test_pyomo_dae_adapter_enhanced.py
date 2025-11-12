import os
import sys
import sympy as sp
import pyomo.environ as pyo

from happymath.Opt.adapters.pyomo_dae_adapter import PyomoDAEAdapter
from happymath.Opt.functional.config import ODEIVPConfig, DomainConfig
from happymath.Opt.opt_expr.processor import ExpressionProcessor


def build_problem_with_param_and_deriv():
    # x'(t) = -a*x(t), x(0)=1, objective min ∫ x'(t)^2 dt (derivatives appear in integrand)
    t = sp.symbols('t', real=True)
    x = sp.Function('x')
    a = sp.symbols('a', real=True)
    ode = [sp.Eq(sp.diff(x(t), t, 1), -a * x(t))]
    func_cfg = ODEIVPConfig(
        ode=ode,
        domain=DomainConfig(var=t, t0=0.0, t1=1.0, grid_n=41),
        ivp_conds={x(0): 1.0},
        constants={},
        control=None,
        objective_meta={0: {"aggregation": "integral", "expr": sp.diff(x(t), t) ** 2}},
        param_symbols=[a],
        param_bounds={a: (0.1, 2.0)},
    )
    ep = ExpressionProcessor()
    pr = ep.process({"min": x(t)**2}, constraints=None, functional_config=func_cfg)
    return pr, func_cfg


def test_pyomo_dae_derivative_and_param():
    pr, func_cfg = build_problem_with_param_and_deriv()
    adapter = PyomoDAEAdapter(pr, func_cfg)
    m = adapter.convert()
    assert isinstance(m, pyo.ConcreteModel)
    # Check that derivative variables and parameter variables exist
    assert any(name.startswith('dx') and name.endswith('_dt') for name, obj in m.component_map(pyo.Var, active=True).items())
    # Parameter variables exist (p_a)
    assert any(name.startswith('p_') for name, obj in m.component_map(pyo.Var, active=True).items())

