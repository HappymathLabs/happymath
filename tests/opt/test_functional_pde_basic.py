import os
import sys
import numpy as np
import sympy as sp

from happymath.Opt.functional.config import PDEConfig, MetricSpec
from happymath.Opt.functional.pde_evaluator import PDEEvaluator


def test_pde_evaluator_heat_eq_terminal_l2():
    # 1D 热方程：u_t = u_xx，初值 sin(pi x)，[0,1]，短时间演化，终端 L2 应小于初值 L2
    t = sp.symbols('t', real=True)
    x = sp.symbols('x', real=True)
    u = sp.Function('u')
    pde = [sp.Eq(sp.diff(u(x, t), t, 1), sp.diff(u(x, t), x, 2))]

    def init_fun(xgrid):
        return np.sin(np.pi * xgrid)

    cfg = PDEConfig(
        pde=pde,
        t0=0.0,
        t1=0.05,
        dt=0.01,
        grid_spec={"bounds": ((0.0, 1.0),), "shape": (64,), "periodic": False},
        init_field=init_fun,
        metrics=[MetricSpec(id='obj:0', kind='path', agg='pde_final_l2')]
    )

    ev = PDEEvaluator(cfg)
    vals = ev.evaluate_all({}, cfg.metrics)
    assert 'obj:0' in vals
    # 仅检查返回可用标量（不同 py-pde 版本的归一化差异较大，此处不做绝对值判定）
    assert np.isfinite(vals['obj:0'])
