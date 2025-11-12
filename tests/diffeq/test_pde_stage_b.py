# -*- coding: utf-8 -*-
"""
Stage B: PDE Extended Capability Tests

Scenarios:
- 2D convection-diffusion (vx, vy scalar coefficients)
- 1D Burgers-like (u*u_x + nu*u_xx)
- 1D wave equation (second-order time derivative replaced with first-order system)
- 2-field Gray-Scott reaction-diffusion (small scale, short time steps)
"""

import numpy as np
import sympy as sp
import pytest

from happymath.DiffEq.PDE.PDEModule import PDEModule
from happymath.DiffEq.PDE.adapters import solve_pde


@pytest.mark.timeout(120)
def test_2d_advection_diffusion_vx_vy():
    x, y, t = sp.symbols('x y t')
    u = sp.Function('u')
    D, vx, vy = sp.symbols('D vx vy')
    expr = sp.Eq(
        sp.Derivative(u(x, y, t), t),
        D*(sp.Derivative(u(x, y, t), (x, 2)) + sp.Derivative(u(x, y, t), (y, 2)))
        - vx*sp.Derivative(u(x, y, t), x)
        - vy*sp.Derivative(u(x, y, t), y)
    )
    m = PDEModule(expr)
    solvable = m.to_solvable_pde
    assert 'd2_dx2(u)' in solvable['u'] and 'd2_dy2(u)' in solvable['u']
    assert 'd_dx(u)' in solvable['u'] and 'd_dy(u)' in solvable['u']

    N = 24
    X, Y = np.meshgrid(
        np.linspace(0, 2*np.pi, N, endpoint=False),
        np.linspace(0, 2*np.pi, N, endpoint=False),
        indexing='ij'
    )
    state = np.sin(X) + np.cos(Y)
    res = solve_pde(
        m,
        state=state,
        t_range=0.001,
        dt=0.0005,
        solver='explicit',
        const_cond={'D': 0.1, 'vx': 0.5, 'vy': -0.3},
        bc='periodic',
        grid_spec={'shape': (N, N), 'bounds': ((0, 1), (0, 1)), 'periodic': True},
    )
    assert res.success


@pytest.mark.timeout(90)
def test_1d_burgers_like():
    x, t = sp.symbols('x t')
    u = sp.Function('u')
    nu = sp.symbols('nu')
    expr = sp.Eq(sp.Derivative(u(x, t), t) + u(x, t)*sp.Derivative(u(x, t), x), nu*sp.Derivative(u(x, t), (x, 2)))
    m = PDEModule(expr)
    solvable = m.to_solvable_pde
    # nu*d2_dx2(u) - u*d_dx(u)
    assert 'd2_dx2(u)' in solvable['u'] and 'u*d_dx(u)' in solvable['u']

    N = 96
    state = np.sin(np.linspace(0, 2*np.pi, N, endpoint=False))
    res = solve_pde(
        m,
        state=state,
        t_range=0.002,
        dt=0.001,
        solver='explicit',
        const_cond={'nu': 0.02},
        bc='periodic',
        grid_spec={'shape': (N,), 'bounds': ((0, 1),), 'periodic': True},
    )
    assert res.success


@pytest.mark.timeout(120)
def test_1d_wave_equation_second_time():
    # u_tt = c^2*u_xx -> forms first-order system through time substitution
    x, t = sp.symbols('x t')
    u = sp.Function('u')
    c = sp.symbols('c')
    expr = sp.Eq(sp.Derivative(u(x, t), (t, 2)), c**2 * sp.Derivative(u(x, t), (x, 2)))
    m = PDEModule(expr)
    solvable = m.to_solvable_pde
    # Should become two-variable system (u and some Y_*)
    assert len(solvable) >= 2
    keys = list(solvable.keys())

    # Construct initial conditions: u=sin(x), first-order time derivative=0
    N = 64
    base = np.sin(np.linspace(0, 2*np.pi, N, endpoint=False))
    # Auto-fill each field: u uses base, other substitute fields use 0
    state = {}
    for k in keys:
        if k == 'u':
            state[k] = base
        else:
            state[k] = np.zeros_like(base)

    res = solve_pde(
        m,
        state=state,
        t_range=0.002,
        dt=0.001,
        solver='explicit',
        const_cond={'c': 1.0},
        bc='periodic',
        grid_spec={'shape': (N,), 'bounds': ((0, 1),), 'periodic': True},
    )
    assert res.success


@pytest.mark.timeout(180)
def test_2field_gray_scott_small():
    # Gray-Scott model (simplified small grid, short time steps)
    x, y, t = sp.symbols('x y t')
    u = sp.Function('u')
    v = sp.Function('v')
    Du, Dv, F, k = sp.symbols('Du Dv F k')
    expr_u = sp.Eq(
        sp.Derivative(u(x, y, t), t),
        Du*(sp.Derivative(u(x, y, t), (x, 2)) + sp.Derivative(u(x, y, t), (y, 2)))
        - u(x, y, t)*v(x, y, t)**2 + F*(1 - u(x, y, t))
    )
    expr_v = sp.Eq(
        sp.Derivative(v(x, y, t), t),
        Dv*(sp.Derivative(v(x, y, t), (x, 2)) + sp.Derivative(v(x, y, t), (y, 2)))
        + u(x, y, t)*v(x, y, t)**2 - (F + k)*v(x, y, t)
    )
    m = PDEModule([expr_u, expr_v])
    solvable = m.to_solvable_pde
    assert set(solvable.keys()) == {'u', 'v'}
    assert 'u*v**2' in solvable['u'] and 'u*v**2' in solvable['v']

    N = 24
    X, Y = np.meshgrid(
        np.linspace(0, 1, N, endpoint=False),
        np.linspace(0, 1, N, endpoint=False),
        indexing='ij'
    )
    # Initial conditions: u close to 1, v close to 0, with small perturbation in the middle
    u0 = np.ones((N, N))
    v0 = np.zeros((N, N))
    cx, cy = N//2, N//2
    u0[cx-2:cx+2, cy-2:cy+2] = 0.50
    v0[cx-2:cx+2, cy-2:cy+2] = 0.25

    state = {'u': u0, 'v': v0}

    res = solve_pde(
        m,
        state=state,
        t_range=0.001,
        dt=0.0005,
        solver='explicit',
        const_cond={'Du': 0.16, 'Dv': 0.08, 'F': 0.035, 'k': 0.065},
        bc='neumann',
        grid_spec={'shape': (N, N), 'bounds': ((0, 1), (0, 1)), 'periodic': False},
    )
    assert res.success

