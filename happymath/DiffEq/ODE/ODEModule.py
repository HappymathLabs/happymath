"""
重构后的ODEModule - 使用组件化架构
将复杂逻辑分离到专门的处理器中，消除全局变量，改善代码结构
"""

import inspect
from collections.abc import Iterable
from IPython.display import display, HTML
from sympy import Function
from typing import Callable, Union, List, Optional, Dict, Any
import numpy as np
import scipy
import scipy.integrate
import sympy
from scipy.integrate import OdeSolver
from scipy.integrate import RK23, RK45, DOP853, Radau, BDF, LSODA
from sympy import dsolve, solve, lambdify
from sympy.core import Number
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.solvers.ode.ode import classify_sysode
from sympy.solvers.ode.systems import _preprocess_eqs
from sympy.utilities.iterables import iterable
from ..diffeq_core.de_base import DEBase
from collections import defaultdict
from functools import partial
import logging
from dataclasses import dataclass

# 导入新的组件
from ..diffeq_expr import process_expression 
from .core.result import ODESolutionResult
from .validators.validators import ParameterValidator
from ..diffeq_core.de_exceptions import (
    DEException, InvalidExpressionError, SolverNotFoundError,
    SolverCreationError, SolverExecutionError, InvalidParameterError,
    MissingParameterError, BoundaryConditionError, ExpressionStandardizationError
)

# 常量定义
DEFAULT_ATOL_MULTIPLIER = 0.001
MAX_SYMBOL_GENERATION_ATTEMPTS = 1000

class ODEModule(DEBase):
    """
    重构后的ODE模块主类
    使用组件化架构，职责分离，线程安全
    """
    
    def __init__(self, sympy_obj: Union[sympy.Expr, list], value_range: str = "real"):
        """
        初始化ODE模块
        
        Args:
            sympy_obj: ODE表达式或表达式列表
            value_range: 变量取值范围
        """
        super(ODEModule, self).__init__(sympy_obj, value_range)
        # expr属性现在是通过property从基类继承的，无需重新设置
        self.range = self._value_range

        # 缓存相关属性
        self._cached_standard_result: Optional[Any] = None
        self._cache_invalid = True
        
        # 设置日志记录器
        self.logger = logging.getLogger(__name__)
        
        # 验证是否为有效的ODE表达式
        if not self.is_ode:
            raise InvalidExpressionError(sympy_obj, "Not a valid ordinary differential equation expression.")
    
    def _compute_standard_ode(self):
        """
        使用 ExprParser 直接计算标准化结果并返回结果对象
        """
        try:
            result = process_expression(self.expr)
            self.undeter_terms = result.undetermined_terms
            # 类型校验：确保为 ODE
            if getattr(result, '_analyzer_result', None) and getattr(result._analyzer_result, 'expression_type', '') != 'ODE':
                raise InvalidExpressionError(self.expr, "Not a valid ordinary differential equation expression.")
            return result
        except Exception as e:
            self.logger.error(f"ODE标准化失败: {e}")
            raise ExpressionStandardizationError(self.expr, "ODE标准化", str(e))
    
    @property
    def stand_ode(self):
        """
        返回标准化后的 ODE 方程列表（使用缓存机制）
        """
        if self._cache_invalid or self._cached_standard_result is None:
            self._cached_standard_result = self._compute_standard_ode()
            self._cache_invalid = False
        return self._cached_standard_result.standardized_expressions
    
    def _invalidate_cache(self) -> None:
        """使缓存失效"""
        self._cache_invalid = True
    
    # 提供必要的属性以供适配器使用（从 ExprParser 结果读取）
    @property
    def Y_symbols(self) -> List[sympy.Symbol]:
        if self._cache_invalid or self._cached_standard_result is None:
            _ = self.stand_ode  # 触发缓存
        try:
            return getattr(self._cached_standard_result, 'Y_symbols', [])
        except Exception:
            return []
    
    @property
    def expr(self) -> Union[sympy.Expr, list]:
        """
        返回表达式对象
        重写基类属性以支持缓存管理
        
        Returns:
            表达式对象
        """
        return self._sympy_obj
    
    @expr.setter
    def expr(self, new_expr: Union[sympy.Expr, list]):
        """
        设置新表达式并自动失效缓存
        修复报告中提到的缓存失效机制不完整问题
        
        Args:
            new_expr: 新的表达式对象
        """
        self._sympy_obj = new_expr
        self._invalidate_cache()
        self.logger.debug("表达式已更新，缓存已失效")
    
    @property
    def show_stand_ode(self) -> None:
        """
        展示标准化后的ODE方程
        优化了异常处理和显示逻辑
        """
        try:
            stand_ode_list = self.stand_ode
            if not stand_ode_list:
                raise ExpressionStandardizationError(self.expr, "系统标准化", "无法转换为标准形式")
            
            print("标准ODE (stand_ode):")
            for stand in stand_ode_list:
                display(stand)
            print("\n")
            
            if hasattr(self, 'subs_vars_dict') and self.subs_vars_dict:
                print("替代变量 (subs_vars_dict):")
                for key, value in self.subs_vars_dict.items():
                    display(sympy.Eq(value, key))
                print("\n")
            
            if hasattr(self, 'undeter_terms') and self.undeter_terms:
                print("待定求解项 (undeter_terms):")
                for und_terms in self.undeter_terms:
                    display(und_terms)
                    
        except Exception as e:
            self.logger.error(f"显示标准ODE失败: {e}")
            raise
    
    def ode2scipy(self, mode: str, cond: Dict, const_cond: Optional[Dict] = None):
        """
        将sympy格式的ODE转换为scipy标准格式
        
        Args:
            mode: 求解模式 ('IVP' 或 'BVP')
            cond: 条件字典
            const_cond: 常数条件字典
            
        Returns:
            scipy格式的函数和参数
        """
        try:
            # 使用包内绝对导入，避免不同导入路径导致的相对导入越界问题
            from happymath.DiffEq.ODE.adapters.ode_scipy_adapter import ode2scipy as _ode2scipy_adapter
            return _ode2scipy_adapter(self, mode, cond, const_cond)
        except Exception as e:
            self.logger.error(f"转换为scipy格式失败: {e}")
            raise SolverExecutionError("ode2scipy", "格式转换", e)
    
    def ana_solve(self, eq: Optional[Union[sympy.Expr, list]] = None, 
                  ics: Optional[Dict] = None, **kwargs) -> Union[sympy.Eq, List[sympy.Eq]]:
        """
        求解析解
        
        Args:
            eq: 方程表达式，默认使用当前表达式
            ics: 初始条件
            **kwargs: 其他dsolve参数
            
        Returns:
            解析解
        """
        if eq is None:
            eq = self.expr
        
        try:
            ana_solution = dsolve(eq, ics=ics, **kwargs)
            return ana_solution
        except Exception as e:
            self.logger.error(f"解析求解失败: {e}")
            raise SolverExecutionError("dsolve", "解析求解", e)
    
    def num_solve(self, mode: str, cond: Dict, domain: np.ndarray, 
                  const_cond: Optional[Dict] = None, bc: Optional[Callable] = None,
                  init_guess: Union[str, np.ndarray] = "linear", 
                  solve_method: str = "RK45",
                  tol: float = 0.001, bc_tol: Optional[float] = None) -> np.ndarray:
        """
        数值求解方法（已合并详细求解逻辑，保持向后兼容：返回解数组）
        
        Args:
            mode: 求解模式
            cond: 条件字典
            domain: 定义域
            const_cond: 常数条件
            bc: 边界条件函数（BVP需要）
            init_guess: 初始猜测
            solve_method: 求解方法
            tol: 容差
            bc_tol: 边界条件容差
            
        Returns:
            解数组（np.ndarray）
        """
        try:
            # 参数校验
            self._validate_solver_parameters(mode, domain, init_guess, bc)

            # 分派求解
            if mode.upper() == "IVP":
                result = self._solve_ivp(cond, domain, solve_method, tol, const_cond)
            elif mode.upper() == "BVP":
                result = self._solve_bvp(cond, domain, bc, init_guess, solve_method, tol, bc_tol, const_cond)
            else:
                raise InvalidParameterError("mode", mode, valid_values=["IVP", "BVP"])

            return result.solution
        except Exception as e:
            self.logger.error(f"数值求解失败: {e}")
            if isinstance(e, DEException):
                raise
            else:
                raise SolverExecutionError(solve_method, "数值求解", e)
    
    def _validate_solver_parameters(self, mode: str, domain: np.ndarray, 
                                  init_guess: Union[str, np.ndarray], 
                                  bc: Optional[Callable]) -> None:
        """
        验证求解器参数
        
        Args:
            mode: 求解模式
            domain: 定义域
            init_guess: 初始猜测
            bc: 边界条件函数
        """
        ParameterValidator.validate_solver_parameters(mode, domain, init_guess, bc)
    
    def _solve_ivp(self, cond: Dict, domain: np.ndarray, solve_method: str,
                   tol: float, const_cond: Optional[Dict] = None) -> ODESolutionResult:
        """
        求解初值问题
        
        Args:
            cond: 条件字典
            domain: 定义域
            solve_method: 求解方法
            tol: 容差
            const_cond: 常数条件
            
        Returns:
            求解结果对象
        """
        try:
            # 转换为scipy格式
            scipy_ode_func, subs_dict_ivp, const_values = self.ode2scipy(
                mode="IVP", cond=cond, const_cond=const_cond
            )
            
            # 使用scipy求解
            sol_ivp = scipy.integrate.solve_ivp(
                fun=scipy_ode_func,
                t_span=[domain[0], domain[-1]],
                t_eval=domain,
                y0=subs_dict_ivp,
                method=solve_method,
                rtol=tol,
                atol=DEFAULT_ATOL_MULTIPLIER * tol
            )
            
            if not sol_ivp.success:
                raise SolverExecutionError(solve_method, "IVP求解", sol_ivp.message)
            
            # 误差估计：简化为容差占位，后续如需可在适配层恢复
            local_errors = [tol] * len(domain)
            
            return ODESolutionResult(
                domain=domain,
                solution=sol_ivp.y.T,
                error=local_errors,
                solution_func=lambda t: sol_ivp.sol(t) if hasattr(sol_ivp, 'sol') else None,
                substitution_dict={"initial_conditions": cond, "constants": const_cond or {}},
                success=True,
                message="IVP求解成功"
            )
            
        except Exception as e:
            if isinstance(e, DEException):
                raise
            else:
                raise SolverExecutionError(solve_method, "IVP求解", e)
    
    def _solve_bvp(self, cond: Dict, domain: np.ndarray, bc: Callable,
                   init_guess: Union[str, np.ndarray], solve_method: str,
                   tol: float, bc_tol: Optional[float] = None,
                   const_cond: Optional[Dict] = None) -> ODESolutionResult:
        """
        求解边值问题
        
        Args:
            cond: 条件字典
            domain: 定义域
            bc: 边界条件函数
            init_guess: 初始猜测
            solve_method: 求解方法
            tol: 容差
            bc_tol: 边界条件容差
            const_cond: 常数条件
            
        Returns:
            求解结果对象
        """
        try:
            if bc is None:
                raise MissingParameterError("bc", "BVP求解")
            
            # 转换为scipy格式
            scipy_ode_func, bc_func, subs_dict_bvp, const_values = self.ode2scipy(
                mode="BVP", cond=cond, const_cond=const_cond
            )
            
            # 处理初始猜测
            if isinstance(init_guess, str) and init_guess == "linear":
                # 生成线性初始猜测
                n_eq = len(self.stand_ode)
                init_y = np.ones((n_eq, len(domain)))
            else:
                init_y = np.array(init_guess)
            
            # 使用scipy.integrate.solve_bvp求解
            if bc_tol is None:
                bc_tol = tol
            
            sol_bvp = scipy.integrate.solve_bvp(
                fun=scipy_ode_func,
                bc=bc_func,  # 使用从适配器返回的边界条件函数
                x=domain,
                y=init_y,
                tol=tol
            )
            
            if not sol_bvp.success:
                raise SolverExecutionError(solve_method, "BVP求解", sol_bvp.message)
            
            # 评估解在域点上的值
            solution_values = sol_bvp.sol(domain).T
            
            return ODESolutionResult(
                domain=domain,
                solution=solution_values,
                error=[tol] * len(domain),  # BVP误差计算较复杂，暂用容差
                solution_func=lambda t: sol_bvp.sol(t),
                substitution_dict={"boundary_conditions": cond, "constants": const_cond or {}},
                success=True,
                message="BVP求解成功"
            )
            
        except Exception as e:
            if isinstance(e, DEException):
                raise
            else:
                raise SolverExecutionError(solve_method, "BVP求解", e)
                
    # 为了与ode_scipy_adapter.py保持兼容性，添加以下方法
    def _stand_ode_der_subs(self, stand_ode_list, Y_symbols):
        """
        根据标准ODE得到导数项的替代项字典
        stand_ode的等号左侧是导数表达式，右侧是替代变量，该函数仅返回所有替代导数项的替代变量
        
        Args:
            stand_ode_list: 标准ODE方程列表
            Y_symbols: Y符号列表
            
        Returns:
            dict: 导数项替代字典
        """
        try:
            Y = Y_symbols
            Y_subs_dict = {}
            used_Y = set()  # 用于记录已经被替代的变量

            for eqs_expr in stand_ode_list:
                for check_Y in Y:
                    if eqs_expr.rhs == check_Y and check_Y not in used_Y:
                        if Y_subs_dict == {}:
                            Y_subs_dict[check_Y] = eqs_expr.lhs
                        else:
                            Y_subs_dict[check_Y] = eqs_expr.lhs.subs(Y_subs_dict)
                        used_Y.add(check_Y)  # 标记该变量已经被替代

            return Y_subs_dict
        except Exception as e:
            self.logger.warning(f"_stand_ode_der_subs失败: {e}")
            return {}
    
    def _select_conds(self, non_derivative_conds, derivative_conds, third_conds):
        """
        将BVP的定解条件按照边界值进行区分
        
        Args:
            non_derivative_conds: 非导数条件
            derivative_conds: 导数条件 
            third_conds: 第三类条件
            
        Returns:
            dict: {ya_value:[non_der_expr1,der_expr1,third_conds_1,...], yb_value:[non_der_expr1,der_expr1,third_conds_1,...]}
        """
        if isinstance(non_derivative_conds, dict):
            non_der_list = [*non_derivative_conds.keys()]  # 函数项定解条件
        elif isinstance(non_derivative_conds, list):
            non_der_list = non_derivative_conds
        else:
            non_der_list = []

        if isinstance(derivative_conds, dict):
            der_list = [*derivative_conds.keys()]  # 导数项定解条件
        elif isinstance(derivative_conds, list):
            der_list = derivative_conds
        else:
            der_list = []

        third_list = third_conds if third_conds else []  # 表达式项定解条件

        selected_conds_dict = defaultdict(list)  # 可变key值的字典
        
        for non_der_key in non_der_list:
            bc_value = self._split_expr_meta(non_der_key)
            if bc_value and self._is_number(bc_value[0]):
                selected_conds_dict[bc_value[0]].append(non_der_key)
            else:
                raise BoundaryConditionError("非导数条件", non_der_key, "边界值条件定义错误")

        for der_key in der_list:
            bc_meta_list = self._split_expr_meta(der_key, mode_list=[sympy.Mul, sympy.Add, sympy.Pow, sympy.Subs])
            bc_value = [y[0] for y in [x for x in bc_meta_list if isinstance(x, sympy.Tuple)] if self._is_number(y[0])]
            if bc_value:
                selected_conds_dict[bc_value[0]].append(der_key)
            else:
                raise BoundaryConditionError("非导数条件", non_der_key, "边界值条件定义错误")

        for third_key in third_list:
            bc_meta_list = self._split_expr_meta(third_key, mode_list=[sympy.Mul, sympy.Add, sympy.Pow, sympy.Subs])
            bc_value = [y[0] for y in [x for x in bc_meta_list if isinstance(x, sympy.Tuple)] if self._is_number(y[0])]
            if bc_value:
                selected_conds_dict[bc_value[0]].append(third_key)
            else:
                raise BoundaryConditionError("非导数条件", non_der_key, "边界值条件定义错误")

        return dict(selected_conds_dict)

    # 确保这些属性存在以保持兼容性
    @property 
    def non_derivative_conds(self):
        if not hasattr(self, '_non_derivative_conds'):
            self._non_derivative_conds = {}
        return self._non_derivative_conds
    
    @non_derivative_conds.setter
    def non_derivative_conds(self, value):
        self._non_derivative_conds = value
    
    @property
    def derivative_conds(self):
        if not hasattr(self, '_derivative_conds'):
            self._derivative_conds = {}
        return self._derivative_conds
    
    @derivative_conds.setter
    def derivative_conds(self, value):
        self._derivative_conds = value
        
    @property
    def org_derivative_conds(self):
        if not hasattr(self, '_org_derivative_conds'):
            self._org_derivative_conds = {}
        return self._org_derivative_conds
    
    @org_derivative_conds.setter
    def org_derivative_conds(self, value):
        self._org_derivative_conds = value
    
    @property
    def third_conds(self):
        if not hasattr(self, '_third_conds'):
            self._third_conds = []
        return self._third_conds
    
    @third_conds.setter
    def third_conds(self, value):
        self._third_conds = value
    
    @property
    def has_const(self):
        if not hasattr(self, '_has_const'):
            self._has_const = None
        return self._has_const
    
    @has_const.setter
    def has_const(self, value):
        self._has_const = value
    
    @property
    def _is_ivp_bvp(self):
        """为ode_scipy_adapter.py兼容性保留的属性"""
        # 这里是一个简化的实现，原来的逻辑更复杂
        if not hasattr(self, '__is_ivp_bvp'):
            self.__is_ivp_bvp = "BVP"  # 默认返回BVP
        return self.__is_ivp_bvp
    
    @_is_ivp_bvp.setter
    def _is_ivp_bvp(self, value):
        self.__is_ivp_bvp = value
