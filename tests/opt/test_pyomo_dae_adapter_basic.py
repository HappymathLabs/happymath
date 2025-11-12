import os
import sys
import sympy as sp
import pyomo.environ as pyo

from happymath.Opt.adapters.pyomo_dae_adapter import PyomoDAEAdapter
from happymath.Opt.functional.config import ODEIVPConfig, DomainConfig
from happymath.Opt.opt_expr.processor import ExpressionProcessor
from happymath.Opt.OptModule import OptModule


def build_problem_without_control():
    """
    Minimal test problem (no control):
      x'(t) = -x(t), x(0)=1, objective min ∫ x(t)^2 dt
    Purpose: Verify DAE adapter can correctly build model, discretize, and set integrated state objectives.
    """
    t = sp.symbols('t', real=True)
    x = sp.Function('x')
    ode = [sp.Eq(sp.diff(x(t), t, 1), -x(t))]
    func_cfg = ODEIVPConfig(
        ode=ode,
        domain=DomainConfig(var=t, t0=0.0, t1=1.0, grid_n=41),
        ivp_conds={x(0): 1.0},
        constants={},
        control=None,
        objective_meta={0: {"aggregation": "integral", "expr": x(t)**2}},
    )
    # Parse (only for adapter construction parse_result)
    ep = ExpressionProcessor()
    pr = ep.process({"min": x(t)**2}, constraints=None, functional_config=func_cfg)
    return pr, func_cfg


def test_pyomo_dae_adapter_builds_and_discretizes():
    pr, func_cfg = build_problem_without_control()
    adapter = PyomoDAEAdapter(pr, func_cfg)
    m = adapter.convert()
    assert isinstance(m, pyo.ConcreteModel)

    # Check that key components exist
    assert hasattr(m, 't')
    assert any(isinstance(obj, pyo.Var) for obj in m.component_objects(pyo.Var, active=True))
    assert any('d' in obj.name for obj in m.component_objects(pyo.Var, active=True))  # dI/dt or derivative variables
    # Check that objective exists
    assert any(isinstance(obj, pyo.Objective) for obj in m.component_objects(pyo.Objective, active=True))


def test_pyomo_solver_builds_dae_model_without_solving():
    pr, func_cfg = build_problem_without_control()
    opt = OptModule({"min": pr.objective_exprs[0]}, None, mode='pyomo', functional_config=func_cfg)
    # Only build model, don't actually solve
    model = opt.pyomo_solver._get_or_create_model()
    assert isinstance(model, pyo.ConcreteModel)
    # Verify discretized (m.t is Set, contains finite points)
    assert len(list(model.t)) > 1
