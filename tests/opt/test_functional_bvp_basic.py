import os
import sys
import sympy as sp

from happymath.Opt.OptModule import OptModule
from happymath.Opt.functional.config import ODEBVPConfig, DomainConfig, MetricSpec
from happymath.Opt.functional.evaluator import TrajectoryEvaluatorBVP


def test_bvp_evaluator_runs_simple():
    # x''(t) = 0, x(0)=0, x(1)=0 => solution is 0; J = terminal x(1) and path L2 are both 0
    t = sp.symbols('t', real=True)
    x = sp.Function('x')
    ode = [sp.Eq(sp.diff(x(t), t, 2), 0)]
    cfg = ODEBVPConfig(
        ode=ode,
        domain=DomainConfig(var=t, t0=0.0, t1=1.0, grid_n=41),
        bvp_conds={x(0): 0.0, x(1): 0.0},
        metrics=[
            MetricSpec(id='obj:0', kind='terminal', state_index=0, agg='terminal'),
            MetricSpec(id='obj:1', kind='path', state_index=0, agg='l2_norm'),
        ],
    )
    ev = TrajectoryEvaluatorBVP(cfg)
    vals = ev.evaluate_all({}, cfg.metrics)
    assert abs(vals['obj:0']) <= 1e-8
    assert abs(vals['obj:1']) <= 1e-8
