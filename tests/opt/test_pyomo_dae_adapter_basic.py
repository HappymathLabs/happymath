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
    测试用最小问题（无控制）：
      x'(t) = -x(t), x(0)=1，目标 min ∫ x(t)^2 dt
    目的：验证 DAE 适配器能正确构建模型、离散化，并设置积分状态目标。
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
    # 解析（仅用于适配器构造 parse_result）
    ep = ExpressionProcessor()
    pr = ep.process({"min": x(t)**2}, constraints=None, functional_config=func_cfg)
    return pr, func_cfg


def test_pyomo_dae_adapter_builds_and_discretizes():
    pr, func_cfg = build_problem_without_control()
    adapter = PyomoDAEAdapter(pr, func_cfg)
    m = adapter.convert()
    assert isinstance(m, pyo.ConcreteModel)

    # 检查关键组件存在
    assert hasattr(m, 't')
    assert any(isinstance(obj, pyo.Var) for obj in m.component_objects(pyo.Var, active=True))
    assert any('d' in obj.name for obj in m.component_objects(pyo.Var, active=True))  # dI/dt 或导数变量
    # 检查目标存在
    assert any(isinstance(obj, pyo.Objective) for obj in m.component_objects(pyo.Objective, active=True))


def test_pyomo_solver_builds_dae_model_without_solving():
    pr, func_cfg = build_problem_without_control()
    opt = OptModule({"min": pr.objective_exprs[0]}, None, mode='pyomo', functional_config=func_cfg)
    # 仅构建模型，不实际求解
    model = opt.pyomo_solver._get_or_create_model()
    assert isinstance(model, pyo.ConcreteModel)
    # 校验已离散化（m.t 为 Set，包含有限点）
    assert len(list(model.t)) > 1
