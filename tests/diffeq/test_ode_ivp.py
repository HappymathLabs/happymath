"""
IVP（初值问题）测试代码
基于ODEModule和scipy的对比测试

测试包括：
1. 一阶ODE的IVP问题
2. 一阶ODE组的IVP问题  
3. 二阶ODE的IVP问题
4. 复杂ODE组的IVP问题
"""

import numpy as np
import pytest
import sympy
from scipy.integrate import solve_ivp

# 导入本地模块
from happymath.DiffEq.ODE.ODEModule import ODEModule


class TestFirstOrderODEIVP:
    """一阶ODE的IVP问题测试类"""
    
    def test_first_order_ode_ivp_case1(self):
        """
        测试案例1: dy/dt = -y + sin(t) - exp(-t) + k + m, y(0) = 0
        包含常数项k=1, m=2的一阶线性ODE
        """
        # 定义sympy符号与函数
        y = sympy.Function("y")
        t = sympy.symbols("t")
        k, m = sympy.symbols("k m", constant=True)

        # 定义ode表达式与初值条件
        ode_expr = -y(t).diff(t, 1) - y(t) - sympy.exp(-t) + sympy.sin(t) + k + m
        ics = {y(0): 0}
        const_cond = {k: 1, m: 2}
        
        # 创建ODEModule对象
        ode_obj = ODEModule(ode_expr)
        
        # 时间范围和初值条件
        t_span = np.linspace(0, 15, 100)
        t_range = (0, 15)
        y0 = [0]

        # scipy求解方式
        def scipy_ode(t, y, k, m):
            return -y - np.exp(-t) + np.sin(t) + k + m

        sol_scipy = solve_ivp(scipy_ode, t_range, y0, t_eval=t_span, args=[1, 2])
        
        # happymath求解方式
        happymath_func, S, const = ode_obj.ode2scipy("IVP", ics, const_cond)
        sol_happymath = solve_ivp(happymath_func, t_range, S, t_eval=t_span, args=const)

        # 比较结果
        assert np.allclose(sol_scipy.y[0], sol_happymath.y[0], rtol=1e-10)

    def test_first_order_ode_ivp_case2(self):
        """
        测试案例2: dy/dt = 2y + t, y(0) = 1
        简单的一阶线性ODE
        """
        y = sympy.Function("y")
        t = sympy.symbols("t")
        
        ode_expr = -y(t).diff(t, 1) + 2*y(t) + t
        ics = {y(0): 1}
        
        ode_obj = ODEModule(ode_expr)
        
        t_span = np.linspace(0, 5, 50)
        t_range = (0, 5)
        y0 = [1]

        def scipy_ode(t, y):
            return 2*y + t

        sol_scipy = solve_ivp(scipy_ode, t_range, y0, t_eval=t_span)
        
        happymath_func, S, const = ode_obj.ode2scipy("IVP", ics)
        sol_happymath = solve_ivp(happymath_func, t_range, S, t_eval=t_span, args=const)

        assert np.allclose(sol_scipy.y[0], sol_happymath.y[0], rtol=1e-10)

    def test_first_order_ode_ivp_case3(self):
        """
        测试案例3: dy/dt = -3y + cos(t), y(0) = 2
        包含三角函数的一阶ODE
        """
        y = sympy.Function("y")
        t = sympy.symbols("t")
        
        ode_expr = -y(t).diff(t, 1) - 3*y(t) + sympy.cos(t)
        ics = {y(0): 2}
        
        ode_obj = ODEModule(ode_expr)
        
        t_span = np.linspace(0, 3, 30)
        t_range = (0, 3)
        y0 = [2]

        def scipy_ode(t, y):
            return -3*y + np.cos(t)

        sol_scipy = solve_ivp(scipy_ode, t_range, y0, t_eval=t_span)
        
        happymath_func, S, const = ode_obj.ode2scipy("IVP", ics)
        sol_happymath = solve_ivp(happymath_func, t_range, S, t_eval=t_span, args=const)

        assert np.allclose(sol_scipy.y[0], sol_happymath.y[0], rtol=1e-10)

    def test_first_order_ode_ivp_case4(self):
        """
        测试案例4: dy/dt = y^2 - t, y(0) = 0.5
        非线性一阶ODE
        """
        y = sympy.Function("y")
        t = sympy.symbols("t")
        
        ode_expr = -y(t).diff(t, 1) + y(t)**2 - t
        ics = {y(0): 0.5}
        
        ode_obj = ODEModule(ode_expr)
        
        t_span = np.linspace(0, 1, 20)
        t_range = (0, 1)
        y0 = [0.5]

        def scipy_ode(t, y):
            return y**2 - t

        sol_scipy = solve_ivp(scipy_ode, t_range, y0, t_eval=t_span)
        
        happymath_func, S, const = ode_obj.ode2scipy("IVP", ics)
        sol_happymath = solve_ivp(happymath_func, t_range, S, t_eval=t_span, args=const)

        assert np.allclose(sol_scipy.y[0], sol_happymath.y[0], rtol=1e-10)


class TestFirstOrderODESystemIVP:
    """一阶ODE组的IVP问题测试类"""
    
    def test_first_order_system_ivp_case1(self):
        """
        测试案例1: dy1/dx = 2*y1 + y2, dy2/dx = y1 + 3*y2
        y1(0) = 1, y2(0) = 0
        """
        y1 = sympy.Function('y1')
        y2 = sympy.Function('y2')
        x = sympy.Symbol('x')

        deqn1 = -y1(x).diff(x, 1) + 2 * y1(x) + y2(x)
        deqn2 = -y2(x).diff(x, 1) + y1(x) + 3 * y2(x)
        deqn = [deqn1, deqn2]
        deqn_obj = ODEModule(deqn)

        t_span = np.linspace(0, 2, 50)
        t_range = (0, 2)
        ics = {y1(0): 1, y2(0): 0}
        y0 = [1, 0]

        def scipy_ode(t, S):
            y1, y2 = S
            return [2 * y1 + y2, y1 + 3 * y2]

        sol_scipy = solve_ivp(scipy_ode, t_range, y0, t_eval=t_span)
        
        happymath_func, S, const = deqn_obj.ode2scipy("IVP", ics)
        sol_happymath = solve_ivp(happymath_func, t_range, S, t_eval=t_span, args=const)

        assert np.allclose(sol_scipy.y[0], sol_happymath.y[0], rtol=1e-10)
        assert np.allclose(sol_scipy.y[1], sol_happymath.y[1], rtol=1e-10)

    def test_first_order_system_ivp_case2(self):
        """
        测试案例2: dy1/dt = -y1 + 2*y2, dy2/dt = 3*y1 - y2
        y1(0) = 2, y2(0) = 1
        """
        y1 = sympy.Function('y1')
        y2 = sympy.Function('y2')
        t = sympy.Symbol('t')

        deqn1 = -y1(t).diff(t, 1) - y1(t) + 2 * y2(t)
        deqn2 = -y2(t).diff(t, 1) + 3 * y1(t) - y2(t)
        deqn = [deqn1, deqn2]
        deqn_obj = ODEModule(deqn)

        t_span = np.linspace(0, 3, 40)
        t_range = (0, 3)
        ics = {y1(0): 2, y2(0): 1}
        y0 = [2, 1]

        def scipy_ode(t, S):
            y1, y2 = S
            return [-y1 + 2*y2, 3*y1 - y2]

        sol_scipy = solve_ivp(scipy_ode, t_range, y0, t_eval=t_span)
        
        happymath_func, S, const = deqn_obj.ode2scipy("IVP", ics)
        sol_happymath = solve_ivp(happymath_func, t_range, S, t_eval=t_span, args=const)

        assert np.allclose(sol_scipy.y[0], sol_happymath.y[0], rtol=1e-10)
        assert np.allclose(sol_scipy.y[1], sol_happymath.y[1], rtol=1e-10)

    def test_first_order_system_ivp_case3(self):
        """
        测试案例3: dy1/dt = y2, dy2/dt = -y1 + sin(t)
        简谐振动系统，y1(0) = 0, y2(0) = 1
        """
        y1 = sympy.Function('y1')
        y2 = sympy.Function('y2')
        t = sympy.Symbol('t')

        deqn1 = -y1(t).diff(t, 1) + y2(t)
        deqn2 = -y2(t).diff(t, 1) - y1(t) + sympy.sin(t)
        deqn = [deqn1, deqn2]
        deqn_obj = ODEModule(deqn)

        t_span = np.linspace(0, 2*np.pi, 60)
        t_range = (0, 2*np.pi)
        ics = {y1(0): 0, y2(0): 1}
        y0 = [0, 1]

        def scipy_ode(t, S):
            y1, y2 = S
            return [y2, -y1 + np.sin(t)]

        sol_scipy = solve_ivp(scipy_ode, t_range, y0, t_eval=t_span)
        
        happymath_func, S, const = deqn_obj.ode2scipy("IVP", ics)
        sol_happymath = solve_ivp(happymath_func, t_range, S, t_eval=t_span, args=const)

        assert np.allclose(sol_scipy.y[0], sol_happymath.y[0], rtol=1e-10)
        assert np.allclose(sol_scipy.y[1], sol_happymath.y[1], rtol=1e-10)


class TestSecondOrderODEIVP:
    """二阶ODE的IVP问题测试类"""
    
    def test_second_order_ode_ivp_case1(self):
        """
        测试案例1: d²y/dt² + y + exp(-t) - sin(t) = 0
        y(0) = 0, y'(0) = 1
        """
        y = sympy.Function("y")
        t = sympy.symbols("t")

        ode_expr = -y(t).diff(t, 2) - y(t) - sympy.exp(-t) + sympy.sin(t)
        ics = {y(0): 0, y(t).diff(t, 1).subs(t, 0): 1}
        
        ode_obj = ODEModule(ode_expr)

        t_span = np.linspace(0, 5, 50)
        t_range = (0, 5)
        y0 = [0, 1]

        def scipy_ode(t, S):
            y, v = S
            return [v, -y - np.exp(-t) + np.sin(t)]

        sol_scipy = solve_ivp(scipy_ode, t_range, y0, t_eval=t_span)
        
        happymath_func, S, const = ode_obj.ode2scipy("IVP", ics)
        sol_happymath = solve_ivp(happymath_func, t_range, S, t_eval=t_span, args=const)

        assert np.allclose(sol_scipy.y[0], sol_happymath.y[0], rtol=1e-10)

    def test_second_order_ode_ivp_case2(self):
        """
        测试案例2: d²y/dt² - 4*dy/dt + 3*y = 0
        y(0) = 1, y'(0) = 0
        """
        y = sympy.Function("y")
        t = sympy.symbols("t")

        ode_expr = -y(t).diff(t, 2) + 4*y(t).diff(t, 1) - 3*y(t)
        ics = {y(0): 1, y(t).diff(t, 1).subs(t, 0): 0}
        
        ode_obj = ODEModule(ode_expr)

        t_span = np.linspace(0, 2, 40)
        t_range = (0, 2)
        y0 = [1, 0]

        def scipy_ode(t, S):
            y, v = S
            return [v, 4*v - 3*y]

        sol_scipy = solve_ivp(scipy_ode, t_range, y0, t_eval=t_span)
        
        happymath_func, S, const = ode_obj.ode2scipy("IVP", ics)
        sol_happymath = solve_ivp(happymath_func, t_range, S, t_eval=t_span, args=const)

        assert np.allclose(sol_scipy.y[0], sol_happymath.y[0], rtol=1e-10)

    def test_second_order_ode_ivp_case3(self):
        """
        测试案例3: d²y/dt² + 2*dy/dt + 2*y = cos(t)
        带阻尼的受迫振动，y(0) = 0, y'(0) = 1
        """
        y = sympy.Function("y")
        t = sympy.symbols("t")

        ode_expr = -y(t).diff(t, 2) - 2*y(t).diff(t, 1) - 2*y(t) + sympy.cos(t)
        ics = {y(0): 0, y(t).diff(t, 1).subs(t, 0): 1}
        
        ode_obj = ODEModule(ode_expr)

        t_span = np.linspace(0, 3, 40)
        t_range = (0, 3)
        y0 = [0, 1]

        def scipy_ode(t, S):
            y, v = S
            return [v, -2*v - 2*y + np.cos(t)]

        sol_scipy = solve_ivp(scipy_ode, t_range, y0, t_eval=t_span)
        
        happymath_func, S, const = ode_obj.ode2scipy("IVP", ics)
        sol_happymath = solve_ivp(happymath_func, t_range, S, t_eval=t_span, args=const)

        assert np.allclose(sol_scipy.y[0], sol_happymath.y[0], rtol=1e-10)


class TestComplexODESystemIVP:
    """复杂ODE组的IVP问题测试类"""
    
    def test_complex_system_ivp_case1(self):
        """
        测试案例1: 混合阶数的复杂ODE组
        d²x1/dt² = 2*x1 + x2, dx2/dt = x1 + 3*x2
        x1(0) = 0, x1'(0) = 1, x2(0) = 0
        """
        y1 = sympy.Function('y1')
        y2 = sympy.Function('y2')
        x = sympy.Symbol('x')

        deqn1 = -y1(x).diff(x, 2) + 2 * y1(x) + y2(x)
        deqn2 = -y2(x).diff(x, 1) + y1(x) + 3 * y2(x)
        deqn = [deqn1, deqn2]
        deqn_obj = ODEModule(deqn)

        t_span = np.linspace(0, 1, 30)
        t_range = (0, 1)
        ics = {y1(x).diff(x).subs(x, 0): 1, y1(0): 0, y2(0): 0}
        y0 = [0, 1, 0]

        def scipy_ode(t, S):
            y1, v, y2 = S
            return [v, 2 * y1 + y2, y1 + 3 * y2]

        sol_scipy = solve_ivp(scipy_ode, t_range, y0, t_eval=t_span)
        
        happymath_func, S, const = deqn_obj.ode2scipy("IVP", ics)
        sol_happymath = solve_ivp(happymath_func, t_range, S, t_eval=t_span, args=const)

        assert np.allclose(sol_scipy.y[0], sol_happymath.y[0], rtol=1e-10)

    def test_complex_system_ivp_case2(self):
        """
        测试案例2: 简化的线性系统
        dx1/dt = x1 + x2, dx2/dt = x2 - x1
        x1(0) = 1, x2(0) = 0
        """
        x1 = sympy.Function("x1")
        x2 = sympy.Function("x2")
        t = sympy.Symbol("t")

        # 简化的线性系统
        ode_1 = -x1(t).diff(t,1) + x1(t) + x2(t)
        ode_2 = -x2(t).diff(t,1) + x2(t) - x1(t)
        ode = [ode_1, ode_2]

        test_obj = ODEModule(ode)
        
        ics_dict = {x1(0): 1, x2(0): 0}

        def scipy_ode(t, S):
            x1, x2 = S
            return [x1 + x2, x2 - x1]

        S_0 = [1, 0]

        t_span = np.linspace(0, 2, 30)
        t_range = (0, 2)

        happymath_func, S, const = test_obj.ode2scipy("IVP", ics_dict)
        sol_happymath = solve_ivp(happymath_func, t_range, S, t_eval=t_span, args=const)

        sol_scipy = solve_ivp(scipy_ode, t_range, S_0, t_eval=t_span)

        assert np.allclose(sol_happymath.y[0], sol_scipy.y[0], rtol=1e-8)
        assert np.allclose(sol_happymath.y[1], sol_scipy.y[1], rtol=1e-8)

    def test_complex_system_ivp_case3(self):
        """
        测试案例3: 三维ODE系统
        dx/dt = -x + y, dy/dt = x - y + z, dz/dt = y - 2*z
        """
        x = sympy.Function('x')
        y = sympy.Function('y')
        z = sympy.Function('z')
        t = sympy.Symbol('t')

        deqn1 = -x(t).diff(t, 1) - x(t) + y(t)
        deqn2 = -y(t).diff(t, 1) + x(t) - y(t) + z(t)
        deqn3 = -z(t).diff(t, 1) + y(t) - 2*z(t)
        deqn = [deqn1, deqn2, deqn3]
        deqn_obj = ODEModule(deqn)

        t_span = np.linspace(0, 2, 30)
        t_range = (0, 2)
        ics = {x(0): 1, y(0): 0, z(0): 0}
        y0 = [1, 0, 0]

        def scipy_ode(t, S):
            x, y, z = S
            return [-x + y, x - y + z, y - 2*z]

        sol_scipy = solve_ivp(scipy_ode, t_range, y0, t_eval=t_span)
        
        happymath_func, S, const = deqn_obj.ode2scipy("IVP", ics)
        sol_happymath = solve_ivp(happymath_func, t_range, S, t_eval=t_span, args=const)

        assert np.allclose(sol_scipy.y[0], sol_happymath.y[0], rtol=1e-10)
        assert np.allclose(sol_scipy.y[1], sol_happymath.y[1], rtol=1e-10)
        assert np.allclose(sol_scipy.y[2], sol_happymath.y[2], rtol=1e-10)


@pytest.fixture
def common_test_parameters():
    """通用测试参数的fixture"""
    return {
        'rtol': 1e-10,
        'default_t_span': np.linspace(0, 2, 30),
        'default_t_range': (0, 2)
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])