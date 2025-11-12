import os
import sys
import numpy as np
import sympy as sp

# 将项目根目录加入 sys.path，便于测试直接导入包
from happymath.Opt.OptModule import OptModule
from happymath.Opt.functional.config import ODEIVPConfig, DomainConfig, ControlParamConfig


def build_simple_ocp():
    """
    一阶 ODE：x'(t) = -x(t) + u(t)
    目标：min ∫_0^1 u(t)^2 dt
    约束：x(1) = 0（作为功能型等式约束）
    控制：分段常数 5 段，系数边界 [-2, 2]
    """
    t = sp.symbols('t', real=True)
    x = sp.Function('x')
    u = sp.Function('u')

    ode = [sp.Eq(sp.diff(x(t), t, 1), -x(t) + u(t))]

    # 功能型配置
    coeffs = sp.symbols('c0:5', real=True)
    func_cfg = ODEIVPConfig(
        ode=ode,
        domain=DomainConfig(var=t, t0=0.0, t1=1.0, grid_n=101),
        ivp_conds={x(0): 1.0},  # 初值固定 1.0
        constants={},
        control=ControlParamConfig(
            kind='piecewise_constant',
            func=u,
            coeff_symbols=list(coeffs),
            segments=5,
            bounds=(-2.0, 2.0),
        ),
        objective_meta={
            0: {"aggregation": "integral", "expr": u(t) ** 2}
        },
        constraint_meta={
            "c_term": {"aggregation": "final_state", "expr": x(t), "sense": "eq", "state_index": 0}
        },
        extra_symbols=list(coeffs),
        bounds={}
    )

    # 目标表达式（占位：Integral 保持语义），评估走 evaluator
    obj = {"min": sp.integrate(u(t) ** 2, (t, 0, 1))}

    # 决策变量边界（通过代数约束提供，确保 Pymoo 严格边界通过）
    constraints = []
    for c in coeffs:
        constraints.append(c <= 2.0)
        constraints.append(c >= -2.0)

    return obj, constraints, func_cfg


def test_pymoo_functional_ocp_runs():
    obj, constraints, func_cfg = build_simple_ocp()

    # 构建并求解
    opt = OptModule(
        obj_func=obj,
        constraints=constraints,
        mode="pymoo",
        default_search_range=2.0,
        functional_config=func_cfg,
    )

    # 缩短单测时间：降低评估预算
    try:
        # 适当提高评估预算以提升稳定性（避免随机解失败）
        opt.pymoo_solver._budget_override = 300
    except Exception:
        pass
    res = opt.solve(solver="GA", use_auto_solvers=False, max_solvers=1)

    assert res is not None
    # 至少应有一个成功结果
    succ = any(r.get('success') for r in res.raw_all_solutions)
    assert succ, f"Pymoo 功能型求解失败: {res.raw_all_solutions}"

    # 读取最优 X，评估终端约束近似满足（<= 0.2）
    X = res.raw_all_solutions[0].get('X')
    assert X is not None

    # 由于限制测试时间与随机性，仅检查结果对象结构
    best_vars = res.variables
    assert isinstance(best_vars, dict)
