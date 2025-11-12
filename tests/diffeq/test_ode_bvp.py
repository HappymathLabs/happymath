"""
BVP (边值问题) 测试模块

本模块包含基于pytest框架的边值问题测试用例，用于验证ODEModule求解BVP问题的正确性。
所有测试都将比较ODEModule求解结果和scipy原生求解结果的一致性。

测试类型包括：
1. 二阶ODE组的BVP问题
2. 带常数的BVP问题  
3. 带第三类边界条件的BVP问题
"""

import numpy as np
import pytest
import sympy
from scipy.integrate import solve_bvp

from happymath.DiffEq.ODE.ODEModule import ODEModule


class TestSecondOrderODESystemBVP:
    """二阶ODE组的BVP问题测试类"""

    def test_second_order_ode_bvp_case1(self):
        """
        测试标准二阶ODE组BVP问题求解
        
        微分方程组:
        -y1''(x) + 2*y1(x) + y2(x) = 0
        -y2'(x) + y1(x) + 3*y2(x) = 0
        
        边界条件:
        y1'(0) = 1, y1(1) = 1, y2(0) = 1
        """
        # 定义函数和变量
        y1 = sympy.Function('y1')
        y2 = sympy.Function('y2')
        x = sympy.Symbol('x')

        # 定义微分方程组
        deqn1 = -y1(x).diff(x, 2) + 2 * y1(x) + y2(x)
        deqn2 = -y2(x).diff(x, 1) + y1(x) + 3 * y2(x)
        deqn = [deqn1, deqn2]
        
        # 创建ODEModule对象
        deqn_obj = ODEModule(deqn)

        # 定义边界条件
        bcs = {y1(x).diff(x).subs(x, 0): 1, y1(1): 1, y2(0): 1}
        
        # 获取happymath求解函数
        happymath_func, bc_func, guess_shape, const_list = deqn_obj.ode2scipy("BVP", bcs)

        # 定义对应的scipy求解函数
        def scipy_ode_second(t, S):
            y1, v, y2 = S
            return [v, 2 * y1 + y2, y1 + 3 * y2]

        def bc(ya, yb):
            return np.array([yb[0] - 1, ya[1] - 1, ya[2] - 1])

        # 设置初值条件
        t_guess = np.linspace(0, 1, 100)
        y_guess = np.zeros((len(guess_shape), t_guess.size))

        # 求解
        sol_scipy_bvp = solve_bvp(scipy_ode_second, bc, t_guess, y_guess)
        sol_happymath_bvp = solve_bvp(happymath_func, bc_func, t_guess, y_guess)

        # 验证结果一致性
        assert np.allclose(sol_scipy_bvp.y[0], sol_happymath_bvp.y[0], rtol=1e-10)
        assert np.allclose(sol_scipy_bvp.y[1], sol_happymath_bvp.y[1], rtol=1e-10)
        assert np.allclose(sol_scipy_bvp.y[2], sol_happymath_bvp.y[2], rtol=1e-10)

    def test_second_order_ode_bvp_case2(self):
        """
        测试不同系数的二阶ODE组BVP问题
        
        微分方程组:
        -y1''(x) + 3*y1(x) + 2*y2(x) = 0
        -y2'(x) + 2*y1(x) + y2(x) = 0
        
        边界条件:
        y1'(0) = 2, y1(1) = 0, y2(0) = -1
        """
        # 定义函数和变量
        y1 = sympy.Function('y1')
        y2 = sympy.Function('y2')
        x = sympy.Symbol('x')

        # 定义微分方程组
        deqn1 = -y1(x).diff(x, 2) + 3 * y1(x) + 2 * y2(x)
        deqn2 = -y2(x).diff(x, 1) + 2 * y1(x) + y2(x)
        deqn = [deqn1, deqn2]
        
        deqn_obj = ODEModule(deqn)

        # 定义边界条件
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
        测试负系数的二阶ODE组BVP问题
        
        微分方程组:
        -y1''(x) - y1(x) + y2(x) = 0
        -y2'(x) + y1(x) - 2*y2(x) = 0
        
        边界条件:
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
        测试更复杂系数的二阶ODE组BVP问题
        
        微分方程组:
        -y1''(x) + 4*y1(x) - 3*y2(x) = 0
        -y2'(x) - 2*y1(x) + 5*y2(x) = 0
        
        边界条件:
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
        测试分数系数的二阶ODE组BVP问题
        
        微分方程组:
        -y1''(x) + 0.5*y1(x) + 1.5*y2(x) = 0
        -y2'(x) + 2.5*y1(x) + 0.8*y2(x) = 0
        
        边界条件:
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
    """带常数的BVP问题测试类"""

    def test_constant_bvp_case1(self):
        """
        测试标准带常数的BVP问题求解
        
        微分方程组:
        -y1''(x) + 2*y1(x) + y2(x) + k = 0
        -y2'(x) + y1(x) + 3*y2(x) = 0
        
        边界条件:
        y1'(0) = 1, y1(1) = 1, y2(0) = 1
        常数条件: k = 0
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
        测试非零常数的BVP问题
        
        微分方程组:
        -y1''(x) + 2*y1(x) + y2(x) + k = 0
        -y2'(x) + y1(x) + 3*y2(x) = 0
        
        常数条件: k = 2.5
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
        测试多常数的BVP问题
        
        微分方程组:
        -y1''(x) + a*y1(x) + y2(x) + b = 0
        -y2'(x) + y1(x) + c*y2(x) = 0
        
        常数条件: a = 1.5, b = -1, c = 2.8
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
        测试负常数的BVP问题
        
        微分方程组:
        -y1''(x) + y1(x) + y2(x) + k = 0
        -y2'(x) + y1(x) + 2*y2(x) = 0
        
        常数条件: k = -3.2
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
    """带第三类边界条件的BVP问题测试类"""

    def test_third_type_bvp_case1(self):
        """
        测试标准第三类边界条件BVP问题
        
        微分方程组:
        -y1''(x) + 2*y1(x) + y2(x) = 0
        -y2'(x) + y1(x) + 3*y2(x) = 0
        
        边界条件:
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
        测试系数不为1的第三类边界条件
        
        边界条件:
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
        测试负系数的第三类边界条件
        
        边界条件:
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
        测试分数系数的第三类边界条件
        
        边界条件:
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
        测试简化的第三类边界条件
        
        边界条件:
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
    """标准求解域fixture"""
    return np.linspace(0, 1, 100)


@pytest.fixture
def extended_domain():
    """扩展求解域fixture"""
    return np.linspace(0, 2, 150)


# 实用工具函数
def create_zero_guess(shape_length, domain_size):
    """创建零初始猜测值"""
    return np.zeros((shape_length, domain_size))


def validate_bvp_solution(scipy_sol, happymath_sol, rtol=1e-10):
    """验证BVP求解结果的一致性"""
    for i in range(len(scipy_sol.y)):
        assert np.allclose(scipy_sol.y[i], happymath_sol.y[i], rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])