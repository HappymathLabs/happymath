"""
BVP (Boundary Value Problem) Test Module

This module contains boundary value problem test cases based on pytest framework, used to verify the correctness of ODEModule for solving BVP problems. All tests will compare the consistency between ODEModule solutions and scipy native solutions.

Test types include:
1. Second-order ODE system BVP problems
2. BVP problems with constants
3. BVP problems with third-type boundary conditions
"""

import numpy as np
import pytest
import sympy
from scipy.integrate import solve_bvp

from happymath.DiffEq.ODE.ODEModule import ODEModule


class TestSecondOrderODESystemBVP:
    """Second-order ODE system BVP problem test class"""

    def test_second_order_ode_bvp_case1(self):
        """
        Test standard second-order ODE system BVP problem solving
        
        Differential equation system:
        -y1''(x) + 2*y1(x) + y2(x) = 0
        -y2'(x) + y1(x) + 3*y2(x) = 0
        
        Boundary conditions:
        y1'(0) = 1, y1(1) = 1, y2(0) = 1
        """
        # Define functions and variables
        y1 = sympy.Function('y1')
        y2 = sympy.Function('y2')
        x = sympy.Symbol('x')

        # Define differential equation system
        deqn1 = -y1(x).diff(x, 2) + 2 * y1(x) + y2(x)
        deqn2 = -y2(x).diff(x, 1) + y1(x) + 3 * y2(x)
        deqn = [deqn1, deqn2]
        
        # Create ODEModule object
        deqn_obj = ODEModule(deqn)

        # Define boundary conditions
        bcs = {y1(x).diff(x).subs(x, 0): 1, y1(1): 1, y2(0): 1}
        
        # Get happymath solving function
        happymath_func, bc_func, guess_shape, const_list = deqn_obj.ode2scipy("BVP", bcs)

        # Define corresponding scipy solving function
        def scipy_ode_second(t, S):
            y1, v, y2 = S
            return [v, 2 * y1 + y2, y1 + 3 * y2]

        def bc(ya, yb):
            return np.array([yb[0] - 1, ya[1] - 1, ya[2] - 1])

        # Set initial conditions
        t_guess = np.linspace(0, 1, 100)
        y_guess = np.zeros((len(guess_shape), t_guess.size))

        # Solve
        sol_scipy_bvp = solve_bvp(scipy_ode_second, bc, t_guess, y_guess)
        sol_happymath_bvp = solve_bvp(happymath_func, bc_func, t_guess, y_guess)

        # Verify result consistency
        assert np.allclose(sol_scipy_bvp.y[0], sol_happymath_bvp.y[0], rtol=1e-10)
        assert np.allclose(sol_scipy_bvp.y[1], sol_happymath_bvp.y[1], rtol=1e-10)
        assert np.allclose(sol_scipy_bvp.y[2], sol_happymath_bvp.y[2], rtol=1e-10)

    def test_second_order_ode_bvp_case2(self):
        """
        Test second-order ODE system BVP problem with different coefficients
        
        Differential equation system:
        -y1''(x) + 3*y1(x) + 2*y2(x) = 0
        -y2'(x) + 2*y1(x) + y2(x) = 0
        
        Boundary conditions:
        y1'(0) = 2, y1(1) = 0, y2(0) = -1
        """
        # Define functions and variables
        y1 = sympy.Function('y1')
        y2 = sympy.Function('y2')
        x = sympy.Symbol('x')

        # Define differential equation system
        deqn1 = -y1(x).diff(x, 2) + 3 * y1(x) + 2 * y2(x)
        deqn2 = -y2(x).diff(x, 1) + 2 * y1(x) + y2(x)
        deqn = [deqn1, deqn2]
        
        deqn_obj = ODEModule(deqn)

        # Define boundary conditions
        bcs = {y1(x).diff(x).subs(x, 0): 2, y1(1): 0, y2(0): -1}
        
        happymath_func, bc_func, guess_shape, const_list = deqn_obj.ode2scipy("BVP", bcs)

        def scipy_ode_second(t, S):
            y1, v, y2 = S
            return [v, 3 * y1 + 2 * y2, 2 * y1 + y2]

        def bc(ya, yb):
            return np.array([yb[0] - 0, ya[1] - 2, ya[2] - (-1)])

        t_guess = np.linspace(0, 1, 100)
        y_guess = np.zeros((len(guess_shape), t_guess.size))

        sol_scipy_bvp = solve_bvp(scipy_ode_second, bc, t_guess, y_guess)
        sol_happymath_bvp = solve_bvp(happymath_func, bc_func, t_guess, y_guess)

        assert np.allclose(sol_scipy_bvp.y[0], sol_happymath_bvp.y[0], rtol=1e-10)
        assert np.allclose(sol_scipy_bvp.y[1], sol_happymath_bvp.y[1], rtol=1e-10)
        assert np.allclose(sol_scipy_bvp.y[2], sol_happymath_bvp.y[2], rtol=1e-10)

    def test_second_order_ode_bvp_case3(self):
        """
        Test second-order ODE system BVP problem with negative coefficients
        
        Differential equation system:
        -y1''(x) - y1(x) + y2(x) = 0
        -y2'(x) + y1(x) - 2*y2(x) = 0
        
        Boundary conditions:
        y1'(0) = 0, y1(1) = 1, y2(0) = 2
        """
        y1 = sympy.Function('y1')
        y2 = sympy.Function('y2')
        x = sympy.Symbol('x')

        deqn1 = -y1(x).diff(x, 2) - y1(x) + y2(x)
        deqn2 = -y2(x).diff(x, 1) + y1(x) - 2 * y2(x)
        deqn = [deqn1, deqn2]
        
        deqn_obj = ODEModule(deqn)

        bcs = {y1(x).diff(x).subs(x, 0): 0, y1(1): 1, y2(0): 2}
        
        happymath_func, bc_func, guess_shape, const_list = deqn_obj.ode2scipy("BVP", bcs)

        def scipy_ode_second(t, S):
            y1, v, y2 = S
            return [v, -y1 + y2, y1 - 2 * y2]

        def bc(ya, yb):
            return np.array([yb[0] - 1, ya[1] - 0, ya[2] - 2])

        t_guess = np.linspace(0, 1, 100)
        y_guess = np.zeros((len(guess_shape), t_guess.size))

        sol_scipy_bvp = solve_bvp(scipy_ode_second, bc, t_guess, y_guess)
        sol_happymath_bvp = solve_bvp(happymath_func, bc_func, t_guess, y_guess)

        assert np.allclose(sol_scipy_bvp.y[0], sol_happymath_bvp.y[0], rtol=1e-10)
        assert np.allclose(sol_scipy_bvp.y[1], sol_happymath_bvp.y[1], rtol=1e-10)
        assert np.allclose(sol_scipy_bvp.y[2], sol_happymath_bvp.y[2], rtol=1e-10)

    def test_second_order_ode_bvp_case4(self):
        """
        Test second-order ODE system BVP problem with more complex coefficients
        
        Differential equation system:
        -y1''(x) + 4*y1(x) - 3*y2(x) = 0
        -y2'(x) - 2*y1(x) + 5*y2(x) = 0
        
        Boundary conditions:
        y1'(0) = -1, y1(1) = 2, y2(0) = 0
        """
        y1 = sympy.Function('y1')
        y2 = sympy.Function('y2')
        x = sympy.Symbol('x')

        deqn1 = -y1(x).diff(x, 2) + 4 * y1(x) - 3 * y2(x)
        deqn2 = -y2(x).diff(x, 1) - 2 * y1(x) + 5 * y2(x)
        deqn = [deqn1, deqn2]
        
        deqn_obj = ODEModule(deqn)

        bcs = {y1(x).diff(x).subs(x, 0): -1, y1(1): 2, y2(0): 0}
        
        happymath_func, bc_func, guess_shape, const_list = deqn_obj.ode2scipy("BVP", bcs)

        def scipy_ode_second(t, S):
            y1, v, y2 = S
            return [v, 4 * y1 - 3 * y2, -2 * y1 + 5 * y2]

        def bc(ya, yb):
            return np.array([yb[0] - 2, ya[1] - (-1), ya[2] - 0])

        t_guess = np.linspace(0, 1, 100)
        y_guess = np.zeros((len(guess_shape), t_guess.size))

        sol_scipy_bvp = solve_bvp(scipy_ode_second, bc, t_guess, y_guess)
        sol_happymath_bvp = solve_bvp(happymath_func, bc_func, t_guess, y_guess)

        assert np.allclose(sol_scipy_bvp.y[0], sol_happymath_bvp.y[0], rtol=1e-10)
        assert np.allclose(sol_scipy_bvp.y[1], sol_happymath_bvp.y[1], rtol=1e-10)
        assert np.allclose(sol_scipy_bvp.y[2], sol_happymath_bvp.y[2], rtol=1e-10)

    def test_second_order_ode_bvp_case5(self):
        """
        Test second-order ODE system BVP problem with fractional coefficients
        
        Differential equation system:
        -y1''(x) + 0.5*y1(x) + 1.5*y2(x) = 0
        -y2'(x) + 2.5*y1(x) + 0.8*y2(x) = 0
        
        Boundary conditions:
        y1'(0) = 1.5, y1(1) = -0.5, y2(0) = 2.2
        """
        y1 = sympy.Function('y1')
        y2 = sympy.Function('y2')
        x = sympy.Symbol('x')

        deqn1 = -y1(x).diff(x, 2) + 0.5 * y1(x) + 1.5 * y2(x)
        deqn2 = -y2(x).diff(x, 1) + 2.5 * y1(x) + 0.8 * y2(x)
        deqn = [deqn1, deqn2]
        
        deqn_obj = ODEModule(deqn)

        bcs = {y1(x).diff(x).subs(x, 0): 1.5, y1(1): -0.5, y2(0): 2.2}
        
        happymath_func, bc_func, guess_shape, const_list = deqn_obj.ode2scipy("BVP", bcs)

        def scipy_ode_second(t, S):
            y1, v, y2 = S
            return [v, 0.5 * y1 + 1.5 * y2, 2.5 * y1 + 0.8 * y2]

        def bc(ya, yb):
            return np.array([yb[0] - (-0.5), ya[1] - 1.5, ya[2] - 2.2])

        t_guess = np.linspace(0, 1, 100)
        y_guess = np.zeros((len(guess_shape), t_guess.size))

        sol_scipy_bvp = solve_bvp(scipy_ode_second, bc, t_guess, y_guess)
        sol_happymath_bvp = solve_bvp(happymath_func, bc_func, t_guess, y_guess)

        assert np.allclose(sol_scipy_bvp.y[0], sol_happymath_bvp.y[0], rtol=1e-10)
        assert np.allclose(sol_scipy_bvp.y[1], sol_happymath_bvp.y[1], rtol=1e-10)
        assert np.allclose(sol_scipy_bvp.y[2], sol_happymath_bvp.y[2], rtol=1e-10)


class TestConstantBVP:
    """BVP problem test class with constants"""

    def test_constant_bvp_case1(self):
        """
        Test standard BVP problem with constants solving
        
        Differential equation system:
        -y1''(x) + 2*y1(x) + y2(x) + k = 0
        -y2'(x) + y1(x) + 3*y2(x) = 0
        
        Boundary conditions:
        y1'(0) = 1, y1(1) = 1, y2(0) = 1
        Constant condition: k = 0
        """
        y1 = sympy.Function('y1')
        y2 = sympy.Function('y2')
        x = sympy.Symbol('x')
        k = sympy.Symbol('k', const=True)

        deqn1 = -y1(x).diff(x, 2) + 2 * y1(x) + y2(x) + k
        deqn2 = -y2(x).diff(x, 1) + y1(x) + 3 * y2(x)
        deqn = [deqn1, deqn2]
        
        deqn_obj = ODEModule(deqn)

        bcs = {y1(x).diff(x).subs(x, 0): 1, y1(1): 1, y2(0): 1}
        const_cond = {k: 0}
        
        happymath_func, bc_func, guess_shape, const_list = deqn_obj.ode2scipy("BVP", bcs, const_cond)

        def scipy_ode_second(t, S):
            y1, v, y2 = S
            return [v, 2 * y1 + y2, y1 + 3 * y2]

        def bc(ya, yb):
            return np.array([yb[0] - 1, ya[1] - 1, ya[2] - 1])

        t_guess = np.linspace(0, 1, 100)
        y_guess = np.zeros((len(guess_shape), t_guess.size))

        sol_scipy_bvp = solve_bvp(scipy_ode_second, bc, t_guess, y_guess)
        sol_happymath_bvp = solve_bvp(happymath_func, bc_func, t_guess, y_guess)

        assert np.allclose(sol_scipy_bvp.y[0], sol_happymath_bvp.y[0], rtol=1e-10)

    def test_constant_bvp_case2(self):
        """
        Test BVP problem with non-zero constants
        
        Differential equation system:
        -y1''(x) + 2*y1(x) + y2(x) + k = 0
        -y2'(x) + y1(x) + 3*y2(x) = 0
        
        Constant condition: k = 2.5
        """
        y1 = sympy.Function('y1')
        y2 = sympy.Function('y2')
        x = sympy.Symbol('x')
        k = sympy.Symbol('k', const=True)

        deqn1 = -y1(x).diff(x, 2) + 2 * y1(x) + y2(x) + k
        deqn2 = -y2(x).diff(x, 1) + y1(x) + 3 * y2(x)
        deqn = [deqn1, deqn2]
        
        deqn_obj = ODEModule(deqn)

        bcs = {y1(x).diff(x).subs(x, 0): 0, y1(1): 0.5, y2(0): -1}
        const_cond = {k: 2.5}
        
        happymath_func, bc_func, guess_shape, const_list = deqn_obj.ode2scipy("BVP", bcs, const_cond)

        def scipy_ode_second(t, S):
            y1, v, y2 = S
            return [v, 2 * y1 + y2 + 2.5, y1 + 3 * y2]

        def bc(ya, yb):
            return np.array([yb[0] - 0.5, ya[1] - 0, ya[2] - (-1)])

        t_guess = np.linspace(0, 1, 100)
        y_guess = np.zeros((len(guess_shape), t_guess.size))

        sol_scipy_bvp = solve_bvp(scipy_ode_second, bc, t_guess, y_guess)
        sol_happymath_bvp = solve_bvp(happymath_func, bc_func, t_guess, y_guess)

        assert np.allclose(sol_scipy_bvp.y[0], sol_happymath_bvp.y[0], rtol=1e-10)

    def test_constant_bvp_case3(self):
        """
        Test BVP problem with multiple constants
        
        Differential equation system:
        -y1''(x) + a*y1(x) + y2(x) + b = 0
        -y2'(x) + y1(x) + c*y2(x) = 0
        
        Constant conditions: a = 1.5, b = -1, c = 2.8
        """
        y1 = sympy.Function('y1')
        y2 = sympy.Function('y2')
        x = sympy.Symbol('x')
        a = sympy.Symbol('a', const=True)
        b = sympy.Symbol('b', const=True)
        c = sympy.Symbol('c', const=True)

        deqn1 = -y1(x).diff(x, 2) + a * y1(x) + y2(x) + b
        deqn2 = -y2(x).diff(x, 1) + y1(x) + c * y2(x)
        deqn = [deqn1, deqn2]
        
        deqn_obj = ODEModule(deqn)

        bcs = {y1(x).diff(x).subs(x, 0): 1, y1(1): 0, y2(0): 1}
        const_cond = {a: 1.5, b: -1, c: 2.8}
        
        happymath_func, bc_func, guess_shape, const_list = deqn_obj.ode2scipy("BVP", bcs, const_cond)

        def scipy_ode_second(t, S):
            y1, v, y2 = S
            return [v, 1.5 * y1 + y2 - 1, y1 + 2.8 * y2]

        def bc(ya, yb):
            return np.array([yb[0] - 0, ya[1] - 1, ya[2] - 1])

        t_guess = np.linspace(0, 1, 100)
        y_guess = np.zeros((len(guess_shape), t_guess.size))

        sol_scipy_bvp = solve_bvp(scipy_ode_second, bc, t_guess, y_guess)
        sol_happymath_bvp = solve_bvp(happymath_func, bc_func, t_guess, y_guess)

        assert np.allclose(sol_scipy_bvp.y[0], sol_happymath_bvp.y[0], rtol=1e-10)

    def test_constant_bvp_case4(self):
        """
        Test BVP problem with negative constants
        
        Differential equation system:
        -y1''(x) + y1(x) + y2(x) + k = 0
        -y2'(x) + y1(x) + 2*y2(x) = 0
        
        Constant condition: k = -3.2
        """
        y1 = sympy.Function('y1')
        y2 = sympy.Function('y2')
        x = sympy.Symbol('x')
        k = sympy.Symbol('k', const=True)

        deqn1 = -y1(x).diff(x, 2) + y1(x) + y2(x) + k
        deqn2 = -y2(x).diff(x, 1) + y1(x) + 2 * y2(x)
        deqn = [deqn1, deqn2]
        
        deqn_obj = ODEModule(deqn)

        bcs = {y1(x).diff(x).subs(x, 0): -0.5, y1(1): 1.2, y2(0): 0.8}
        const_cond = {k: -3.2}
        
        happymath_func, bc_func, guess_shape, const_list = deqn_obj.ode2scipy("BVP", bcs, const_cond)

        def scipy_ode_second(t, S):
            y1, v, y2 = S
            return [v, y1 + y2 - 3.2, y1 + 2 * y2]

        def bc(ya, yb):
            return np.array([yb[0] - 1.2, ya[1] - (-0.5), ya[2] - 0.8])

        t_guess = np.linspace(0, 1, 100)
        y_guess = np.zeros((len(guess_shape), t_guess.size))

        sol_scipy_bvp = solve_bvp(scipy_ode_second, bc, t_guess, y_guess)
        sol_happymath_bvp = solve_bvp(happymath_func, bc_func, t_guess, y_guess)

        assert np.allclose(sol_scipy_bvp.y[0], sol_happymath_bvp.y[0], rtol=1e-10)


class TestThirdTypeBoundaryConditionBVP:
    """BVP problem test class with third-type boundary conditions"""

    def test_third_type_bvp_case1(self):
        """
        Test standard third-type boundary condition BVP problem
        
        Differential equation system:
        -y1''(x) + 2*y1(x) + y2(x) = 0
        -y2'(x) + y1(x) + 3*y2(x) = 0
        
        Boundary conditions:
        y1'(0) + y2(0) = 1, y1(1) = 1, y2(1) = 1
        """
        y1 = sympy.Function('y1')
        y2 = sympy.Function('y2')
        x = sympy.Symbol('x')

        deqn1 = -y1(x).diff(x, 2) + 2 * y1(x) + y2(x)
        deqn2 = -y2(x).diff(x, 1) + y1(x) + 3 * y2(x)
        deqn = [deqn1, deqn2]
        
        deqn_obj = ODEModule(deqn)

        bcs = {y1(x).diff(x).subs(x, 0) + y2(0): 1, y1(1): 1, y2(1): 1}
        
        happymath_func, bc_func, guess_shape, const_list = deqn_obj.ode2scipy("BVP", bcs)

        def scipy_ode_second(t, S):
            y1, v, y2 = S
            return [v, 2 * y1 + y2, y1 + 3 * y2]

        def bc(ya, yb):
            return np.array([yb[0] - 1, ya[1] + ya[2] - 1, yb[2] - 1])

        t_guess = np.linspace(0, 1, 100)
        y_guess = np.zeros((len(guess_shape), t_guess.size))

        sol_scipy_bvp = solve_bvp(scipy_ode_second, bc, t_guess, y_guess)
        sol_happymath_bvp = solve_bvp(happymath_func, bc_func, t_guess, y_guess)

        assert np.allclose(sol_scipy_bvp.y[0], sol_happymath_bvp.y[0], rtol=1e-10)

    def test_third_type_bvp_case2(self):
        """
        Test third-type boundary condition with coefficients not equal to 1
        
        Boundary conditions:
        2*y1'(0) + 3*y2(0) = 5, y1(1) = 0, y2(1) = 2
        """
        y1 = sympy.Function('y1')
        y2 = sympy.Function('y2')
        x = sympy.Symbol('x')

        deqn1 = -y1(x).diff(x, 2) + y1(x) + 2 * y2(x)
        deqn2 = -y2(x).diff(x, 1) + 3 * y1(x) + y2(x)
        deqn = [deqn1, deqn2]
        
        deqn_obj = ODEModule(deqn)

        bcs = {2*y1(x).diff(x).subs(x, 0) + 3*y2(0): 5, y1(1): 0, y2(1): 2}
        
        happymath_func, bc_func, guess_shape, const_list = deqn_obj.ode2scipy("BVP", bcs)

        def scipy_ode_second(t, S):
            y1, v, y2 = S
            return [v, y1 + 2 * y2, 3 * y1 + y2]

        def bc(ya, yb):
            return np.array([yb[0] - 0, 2*ya[1] + 3*ya[2] - 5, yb[2] - 2])

        t_guess = np.linspace(0, 1, 100)
        y_guess = np.zeros((len(guess_shape), t_guess.size))

        sol_scipy_bvp = solve_bvp(scipy_ode_second, bc, t_guess, y_guess)
        sol_happymath_bvp = solve_bvp(happymath_func, bc_func, t_guess, y_guess)

        assert np.allclose(sol_scipy_bvp.y[0], sol_happymath_bvp.y[0], rtol=1e-10)

    def test_third_type_bvp_case3(self):
        """
        Test third-type boundary condition with negative coefficients
        
        Boundary conditions:
        y1'(0) - 2*y2(0) = -1, y1(1) = 1.5, y2(1) = -0.5
        """
        y1 = sympy.Function('y1')
        y2 = sympy.Function('y2')
        x = sympy.Symbol('x')

        deqn1 = -y1(x).diff(x, 2) + 1.5 * y1(x) + y2(x)
        deqn2 = -y2(x).diff(x, 1) + y1(x) + 2.5 * y2(x)
        deqn = [deqn1, deqn2]
        
        deqn_obj = ODEModule(deqn)

        bcs = {y1(x).diff(x).subs(x, 0) - 2*y2(0): -1, y1(1): 1.5, y2(1): -0.5}
        
        happymath_func, bc_func, guess_shape, const_list = deqn_obj.ode2scipy("BVP", bcs)

        def scipy_ode_second(t, S):
            y1, v, y2 = S
            return [v, 1.5 * y1 + y2, y1 + 2.5 * y2]

        def bc(ya, yb):
            return np.array([yb[0] - 1.5, ya[1] - 2*ya[2] - (-1), yb[2] - (-0.5)])

        t_guess = np.linspace(0, 1, 100)
        y_guess = np.zeros((len(guess_shape), t_guess.size))

        sol_scipy_bvp = solve_bvp(scipy_ode_second, bc, t_guess, y_guess)
        sol_happymath_bvp = solve_bvp(happymath_func, bc_func, t_guess, y_guess)

        assert np.allclose(sol_scipy_bvp.y[0], sol_happymath_bvp.y[0], rtol=1e-10)

    def test_third_type_bvp_case4(self):
        """
        Test third-type boundary condition with fractional coefficients
        
        Boundary conditions:
        0.5*y1'(0) + 1.2*y2(0) = 2.3, y1(1) = 0.8, y2(1) = 1.1
        """
        y1 = sympy.Function('y1')
        y2 = sympy.Function('y2')
        x = sympy.Symbol('x')

        deqn1 = -y1(x).diff(x, 2) + 0.8 * y1(x) + 1.3 * y2(x)
        deqn2 = -y2(x).diff(x, 1) + 1.5 * y1(x) + 0.9 * y2(x)
        deqn = [deqn1, deqn2]
        
        deqn_obj = ODEModule(deqn)

        bcs = {0.5*y1(x).diff(x).subs(x, 0) + 1.2*y2(0): 2.3, y1(1): 0.8, y2(1): 1.1}
        
        happymath_func, bc_func, guess_shape, const_list = deqn_obj.ode2scipy("BVP", bcs)

        def scipy_ode_second(t, S):
            y1, v, y2 = S
            return [v, 0.8 * y1 + 1.3 * y2, 1.5 * y1 + 0.9 * y2]

        def bc(ya, yb):
            return np.array([yb[0] - 0.8, 0.5*ya[1] + 1.2*ya[2] - 2.3, yb[2] - 1.1])

        t_guess = np.linspace(0, 1, 100)
        y_guess = np.zeros((len(guess_shape), t_guess.size))

        sol_scipy_bvp = solve_bvp(scipy_ode_second, bc, t_guess, y_guess)
        sol_happymath_bvp = solve_bvp(happymath_func, bc_func, t_guess, y_guess)

        assert np.allclose(sol_scipy_bvp.y[0], sol_happymath_bvp.y[0], rtol=1e-10)

    def test_third_type_bvp_case5(self):
        """
        Test simplified third-type boundary condition
        
        Boundary conditions:
        y1'(0) + y2(0) = 2, y1(1) = 0, y2(1) = 1
        """
        y1 = sympy.Function('y1')
        y2 = sympy.Function('y2')
        x = sympy.Symbol('x')

        deqn1 = -y1(x).diff(x, 2) + 2 * y1(x) + y2(x)
        deqn2 = -y2(x).diff(x, 1) + y1(x) + 3 * y2(x)
        deqn = [deqn1, deqn2]
        
        deqn_obj = ODEModule(deqn)

        bcs = {y1(x).diff(x).subs(x, 0) + y2(0): 2, y1(1): 0, y2(1): 1}
        
        happymath_func, bc_func, guess_shape, const_list = deqn_obj.ode2scipy("BVP", bcs)

        def scipy_ode_second(t, S):
            y1, v, y2 = S
            return [v, 2 * y1 + y2, y1 + 3 * y2]

        def bc(ya, yb):
            return np.array([yb[0] - 0, ya[1] + ya[2] - 2, yb[2] - 1])

        t_guess = np.linspace(0, 1, 100)
        y_guess = np.zeros((len(guess_shape), t_guess.size))

        sol_scipy_bvp = solve_bvp(scipy_ode_second, bc, t_guess, y_guess)
        sol_happymath_bvp = solve_bvp(happymath_func, bc_func, t_guess, y_guess)

        assert np.allclose(sol_scipy_bvp.y[0], sol_happymath_bvp.y[0], rtol=1e-10)


# pytest fixtures for common setup
@pytest.fixture
def standard_domain():
    """Standard solution domain fixture"""
    return np.linspace(0, 1, 100)


@pytest.fixture
def extended_domain():
    """Extended solution domain fixture"""
    return np.linspace(0, 2, 150)


# Utility functions
def create_zero_guess(shape_length, domain_size):
    """Create zero initial guess values"""
    return np.zeros((shape_length, domain_size))


def validate_bvp_solution(scipy_sol, happymath_sol, rtol=1e-10):
    """Validate consistency of BVP solution results"""
    for i in range(len(scipy_sol.y)):
        assert np.allclose(scipy_sol.y[i], happymath_sol.y[i], rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])