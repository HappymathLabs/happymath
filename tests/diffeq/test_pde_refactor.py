# -*- coding: utf-8 -*-
# 基础PDE功能回归测试（第一阶段重构后）

import numpy as np
import sympy as sp
import pytest

from happymath.DiffEq.PDE.PDEModule import PDEModule
from happymath.DiffEq.PDE.adapters import solve_pde


@pytest.mark.timeout(60)
def test_1d_advection_diffusion_periodic():
    x, t = sp.symbols('x t')
    u = sp.Function('u')
    D, v = sp.symbols('D v')
    expr = sp.Eq(sp.Derivative(u(x, t), t), D*sp.Derivative(u(x, t), (x, 2)) - v*sp.Derivative(u(x, t), x))
    m = PDEModule(expr)
    # 检查字符串RHS
    solvable = m.to_solvable_pde
    assert 'u' in solvable and 'd2_dx2(u)' in solvable['u'] and 'd_dx(u)' in solvable['u']
    # 数值求解（周期）
    N = 64
    state = np.sin(np.linspace(0, 2*np.pi, N, endpoint=False))
    res = solve_pde(
        m,
        state=state,
        t_range=0.002,
        dt=0.001,
        solver='explicit',
        const_cond={'D': 0.1, 'v': 0.5},
        bc='periodic',
        grid_spec={'shape': (N,), 'bounds': ((0, 1),), 'periodic': True},
    )
    assert res.success and isinstance(res.rhs, dict)


@pytest.mark.timeout(60)
def test_1d_reaction_diffusion_periodic():
    x, t = sp.symbols('x t')
    u = sp.Function('u')
    D, r = sp.symbols('D r')
    expr = sp.Eq(sp.Derivative(u(x, t), t), D*sp.Derivative(u(x, t), (x, 2)) + r*u(x, t)*(1 - u(x, t)))
    m = PDEModule(expr)
    solvable = m.to_solvable_pde
    # 反应项应无 u(x,t) 调用
    assert 'u**2' in solvable['u'] and 'u(x, t)' not in solvable['u']

    N = 64
    state = np.sin(np.linspace(0, 2*np.pi, N, endpoint=False))
    res = solve_pde(
        m,
        state=state,
        t_range=0.002,
        dt=0.001,
        solver='explicit',
        const_cond={'D': 0.1, 'r': 1.0},
        bc='periodic',
        grid_spec={'shape': (N,), 'bounds': ((0, 1),), 'periodic': True},
    )
    assert res.success


@pytest.mark.timeout(60)
def test_2d_isotropic_diffusion_periodic():
    x, y, t = sp.symbols('x y t')
    u = sp.Function('u')
    a = sp.symbols('a')
    expr = sp.Eq(sp.Derivative(u(x, y, t), t), a*(sp.Derivative(u(x, y, t), (x, 2)) + sp.Derivative(u(x, y, t), (y, 2))))
    m = PDEModule(expr)
    solvable = m.to_solvable_pde
    # 应为 d2_dx2 与 d2_dy2 组合
    assert 'd2_dx2(u)' in solvable['u'] and 'd2_dy2(u)' in solvable['u']

    N = 32
    X, Y = np.meshgrid(np.linspace(0, 2*np.pi, N, endpoint=False), np.linspace(0, 2*np.pi, N, endpoint=False), indexing='ij')
    state = np.sin(X) + np.cos(Y)
    res = solve_pde(
        m,
        state=state,
        t_range=0.002,
        dt=0.001,
        solver='explicit',
        const_cond={'a': 1.0},
        bc='periodic',
        grid_spec={'shape': (N, N), 'bounds': ((0, 1), (0, 1)), 'periodic': True},
    )
    assert res.success


@pytest.mark.timeout(60)
def test_2d_mixed_derivative_periodic():
    x, y, t = sp.symbols('x y t')
    u = sp.Function('u')
    expr = sp.Eq(sp.Derivative(u(x, y, t), t), sp.Derivative(u(x, y, t), x, y))
    m = PDEModule(expr)
    solvable = m.to_solvable_pde
    # 应被映射成嵌套一阶导数
    assert solvable['u'] == 'd_dy(d_dx(u))' or solvable['u'] == 'd_dx(d_dy(u))'

    N = 32
    X, Y = np.meshgrid(np.linspace(0, 2*np.pi, N, endpoint=False), np.linspace(0, 2*np.pi, N, endpoint=False), indexing='ij')
    state = np.sin(X) + np.cos(Y)
    res = solve_pde(
        m,
        state=state,
        t_range=0.002,
        dt=0.001,
        solver='explicit',
        const_cond={},
        bc='periodic',
        grid_spec={'shape': (N, N), 'bounds': ((0, 1), (0, 1)), 'periodic': True},
    )
    assert res.success


@pytest.mark.timeout(60)
def test_1d_coupled_two_fields():
    x, t = sp.symbols('x t')
    u = sp.Function('u')
    vfun = sp.Function('v')
    D1, D2 = sp.symbols('D1 D2')
    expr_u = sp.Eq(sp.Derivative(u(x, t), t), D1*sp.Derivative(u(x, t), (x, 2)) + vfun(x, t))
    expr_v = sp.Eq(sp.Derivative(vfun(x, t), t), D2*sp.Derivative(vfun(x, t), (x, 2)) - u(x, t))
    m = PDEModule([expr_u, expr_v])
    solvable = m.to_solvable_pde
    # 多场名应为 'u','v'
    assert set(solvable.keys()) == {'u', 'v'}
    assert 'v' in solvable['u'] and 'u' in solvable['v']

    N = 64
    state = {
        'u': np.sin(np.linspace(0, 2*np.pi, N, endpoint=False)),
        'v': np.cos(np.linspace(0, 2*np.pi, N, endpoint=False)),
    }
    res = solve_pde(
        m,
        state=state,
        t_range=0.004,
        dt=0.002,
        solver='explicit',
        const_cond={'D1': 0.1, 'D2': 0.05},
        bc='periodic',
        grid_spec={'shape': (N,), 'bounds': ((0, 1),), 'periodic': True},
    )
    assert res.success
