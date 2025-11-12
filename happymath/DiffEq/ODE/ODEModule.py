"""
Refactored ODEModule using a componentized architecture.
Complex logic is separated into dedicated processors; no global state; improved structure.
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
    Main ODE module class (refactored).
    Componentized design with clear responsibilities; thread-safe.
    """
    
    def __init__(self, sympy_obj: Union[sympy.Expr, list], value_range: str = "real"):
        """
        Initialize the ODE module.

        Args:
            sympy_obj: A SymPy ODE expression or list of expressions.
            value_range: Symbol assumptions for variables (e.g., "real").

        Example:
            >>> from sympy import Eq, Function, symbols, Derivative
            >>> t = symbols("t")
            >>> y = Function("y")
            >>> ode = Eq(Derivative(y(t), t), -y(t))
            >>> mod = ODEModule(ode)
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
        """Compute and cache standardized ODE expressions using the expression parser."""
        try:
            result = process_expression(self.expr)
            self.undeter_terms = result.undetermined_terms
            # 类型校验：确保为 ODE
            if getattr(result, '_analyzer_result', None) and getattr(result._analyzer_result, 'expression_type', '') != 'ODE':
                raise InvalidExpressionError(self.expr, "Not a valid ordinary differential equation expression.")
            return result
        except Exception as e:
            self.logger.error(f"Failed to standardize ODE: {e}")
            raise ExpressionStandardizationError(self.expr, "ODE standardization", str(e))
    
    @property
    def stand_ode(self):
        """Return standardized ODE expressions (with caching)."""
        if self._cache_invalid or self._cached_standard_result is None:
            self._cached_standard_result = self._compute_standard_ode()
            self._cache_invalid = False
        return self._cached_standard_result.standardized_expressions
    
    def _invalidate_cache(self) -> None:
        """Invalidate cached standardized result."""
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
        Return the current expression object.
        Override to integrate cache invalidation.

        Returns:
            The SymPy expression or list of expressions.
        """
        return self._sympy_obj
    
    @expr.setter
    def expr(self, new_expr: Union[sympy.Expr, list]):
        """
        Set a new expression and invalidate internal caches.

        Args:
            new_expr: New SymPy expression or list of expressions.
        """
        self._sympy_obj = new_expr
        self._invalidate_cache()
        self.logger.debug("Expression updated; cache invalidated")
    
    @property
    def show_stand_ode(self) -> None:
        """
        Display standardized ODE expressions.
        Includes better error handling and formatting.
        """
        try:
            stand_ode_list = self.stand_ode
            if not stand_ode_list:
                raise ExpressionStandardizationError(self.expr, "system standardization", "cannot convert to standard form")
            
            print("Standard ODE (stand_ode):")
            for stand in stand_ode_list:
                display(stand)
            print("\\n")
            
            if hasattr(self, 'subs_vars_dict') and self.subs_vars_dict:
                print("Substitution variables (subs_vars_dict):")
                for key, value in self.subs_vars_dict.items():
                    display(sympy.Eq(value, key))
                print("\\n")
            
            if hasattr(self, 'undeter_terms') and self.undeter_terms:
                print("Undetermined terms (undeter_terms):")
                for und_terms in self.undeter_terms:
                    display(und_terms)
                    
        except Exception as e:
            self.logger.error(f"Failed to display standardized ODE: {e}")
            raise
    
    def ode2scipy(self, mode: str, cond: Dict, const_cond: Optional[Dict] = None):
        """
        Convert a SymPy ODE into SciPy-compatible callable(s).

        Args:
            mode: Solve mode ('IVP' or 'BVP').
            cond: Conditions dictionary (initial or boundary conditions).
            const_cond: Optional constants dictionary.

        Returns:
            Tuple of SciPy-compatible functions and parameters.
        """
        try:
            # 使用包内绝对导入，避免不同导入路径导致的相对导入越界问题
            from happymath.DiffEq.ODE.adapters.ode_scipy_adapter import ode2scipy as _ode2scipy_adapter
            return _ode2scipy_adapter(self, mode, cond, const_cond)
        except Exception as e:
            self.logger.error(f"Failed to convert to SciPy format: {e}")
            raise SolverExecutionError("ode2scipy", "format conversion", e)
    
    def ana_solve(self, eq: Optional[Union[sympy.Expr, list]] = None, 
                  ics: Optional[Dict] = None, **kwargs) -> Union[sympy.Eq, List[sympy.Eq]]:
        """
        Compute the analytic solution using SymPy dsolve.

        Args:
            eq: Optional equation/expression; default to current expression.
            ics: Initial conditions for dsolve.
            **kwargs: Extra keyword arguments forwarded to dsolve.

        Returns:
            SymPy Eq or list of Eq representing the solution.
        """
        if eq is None:
            eq = self.expr
        
        try:
            ana_solution = dsolve(eq, ics=ics, **kwargs)
            return ana_solution
        except Exception as e:
            self.logger.error(f"Analytic solve failed: {e}")
            raise SolverExecutionError("dsolve", "analytic solve", e)
    
    def num_solve(self, mode: str, cond: Dict, domain: np.ndarray, 
                  const_cond: Optional[Dict] = None, bc: Optional[Callable] = None,
                  init_guess: Union[str, np.ndarray] = "linear", 
                  solve_method: str = "RK45",
                  tol: float = 0.001, bc_tol: Optional[float] = None) -> np.ndarray:
        """
        Numerical solver entry point (returns solution array for backward compatibility).

        Args:
            mode: Solve mode ('IVP' or 'BVP').
            cond: Conditions dictionary.
            domain: Independent variable sample points (numpy array).
            const_cond: Optional constants dictionary.
            bc: Boundary condition function (required for BVP).
            init_guess: Initial guess strategy or array for BVP.
            solve_method: Underlying SciPy method name.
            tol: Relative tolerance.
            bc_tol: Boundary condition tolerance (optional).

        Returns:
            Solution array as numpy.ndarray.
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
            self.logger.error(f"Numeric solve failed: {e}")
            if isinstance(e, DEException):
                raise
            else:
                raise SolverExecutionError(solve_method, "numeric solve", e)
    
    def _validate_solver_parameters(self, mode: str, domain: np.ndarray, 
                                  init_guess: Union[str, np.ndarray], 
                                  bc: Optional[Callable]) -> None:
        """
        Validate common solver parameters.

        Args:
            mode: Solve mode.
            domain: Domain array.
            init_guess: Initial guess setting.
            bc: Boundary condition function.
        """
        ParameterValidator.validate_solver_parameters(mode, domain, init_guess, bc)
    
    def _solve_ivp(self, cond: Dict, domain: np.ndarray, solve_method: str,
                   tol: float, const_cond: Optional[Dict] = None) -> ODESolutionResult:
        """
        Solve an initial value problem (IVP).

        Args:
            cond: Initial conditions mapping.
            domain: 1D domain array (time grid).
            solve_method: SciPy method name.
            tol: Relative tolerance.
            const_cond: Optional constants mapping.

        Returns:
            ODESolutionResult
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
                raise SolverExecutionError(solve_method, "IVP solve", sol_ivp.message)
            
            # 误差估计：简化为容差占位，后续如需可在适配层恢复
            local_errors = [tol] * len(domain)
            
            return ODESolutionResult(
                domain=domain,
                solution=sol_ivp.y.T,
                error=local_errors,
                solution_func=lambda t: sol_ivp.sol(t) if hasattr(sol_ivp, 'sol') else None,
                substitution_dict={"initial_conditions": cond, "constants": const_cond or {}},
                success=True,
                message="IVP solve succeeded"
            )
            
        except Exception as e:
            if isinstance(e, DEException):
                raise
            else:
                raise SolverExecutionError(solve_method, "IVP solve", e)
    
    def _solve_bvp(self, cond: Dict, domain: np.ndarray, bc: Callable,
                   init_guess: Union[str, np.ndarray], solve_method: str,
                   tol: float, bc_tol: Optional[float] = None,
                   const_cond: Optional[Dict] = None) -> ODESolutionResult:
        """
        Solve a boundary value problem (BVP).

        Args:
            cond: Boundary conditions mapping.
            domain: 1D domain array (grid).
            bc: Boundary condition function.
            init_guess: Initial guess setting or array.
            solve_method: Underlying method name.
            tol: Relative tolerance.
            bc_tol: Boundary condition tolerance.
            const_cond: Optional constants mapping.

        Returns:
            ODESolutionResult
        """
        try:
            if bc is None:
                raise MissingParameterError("bc", "BVP solve")
            
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
                raise SolverExecutionError(solve_method, "BVP solve", sol_bvp.message)
            
            # 评估解在域点上的值
            solution_values = sol_bvp.sol(domain).T
            
            return ODESolutionResult(
                domain=domain,
                solution=solution_values,
                error=[tol] * len(domain),  # BVP误差计算较复杂，暂用容差
                solution_func=lambda t: sol_bvp.sol(t),
                substitution_dict={"boundary_conditions": cond, "constants": const_cond or {}},
                success=True,
                message="BVP solve succeeded"
            )
            
        except Exception as e:
            if isinstance(e, DEException):
                raise
            else:
                raise SolverExecutionError(solve_method, "BVP solve", e)
                
    # Compatibility helpers for ode_scipy_adapter.py
    def _stand_ode_der_subs(self, stand_ode_list, Y_symbols):
        """
        Build a mapping for derivative substitutions from standardized ODEs.

        The left-hand side of each standardized equation is the derivative; the right-hand side
        is the substituted symbol. This function returns a dict for all substituted derivative terms.

        Args:
            stand_ode_list: List of standardized SymPy equations.
            Y_symbols: Symbols used for substitutions.

        Returns:
            dict: Derivative substitution mapping.
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
            self.logger.warning(f"_stand_ode_der_subs failed: {e}")
            return {}
    
    def _select_conds(self, non_derivative_conds, derivative_conds, third_conds):
        """
        Group BVP conditions by boundary value.

        Args:
            non_derivative_conds: Non-derivative conditions.
            derivative_conds: Derivative conditions.
            third_conds: Third-type (expression) conditions.

        Returns:
            dict: {ya_value: [...], yb_value: [...]} mapping.
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
                raise BoundaryConditionError("non-derivative-condition", non_der_key, "invalid boundary value condition")

        for der_key in der_list:
            bc_meta_list = self._split_expr_meta(der_key, mode_list=[sympy.Mul, sympy.Add, sympy.Pow, sympy.Subs])
            bc_value = [y[0] for y in [x for x in bc_meta_list if isinstance(x, sympy.Tuple)] if self._is_number(y[0])]
            if bc_value:
                selected_conds_dict[bc_value[0]].append(der_key)
            else:
                raise BoundaryConditionError("non-derivative-condition", non_der_key, "invalid boundary value condition")

        for third_key in third_list:
            bc_meta_list = self._split_expr_meta(third_key, mode_list=[sympy.Mul, sympy.Add, sympy.Pow, sympy.Subs])
            bc_value = [y[0] for y in [x for x in bc_meta_list if isinstance(x, sympy.Tuple)] if self._is_number(y[0])]
            if bc_value:
                selected_conds_dict[bc_value[0]].append(third_key)
            else:
                raise BoundaryConditionError("non-derivative-condition", non_der_key, "invalid boundary value condition")

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
        """Compatibility attribute kept for ode_scipy_adapter.py."""
        # 这里是一个简化的实现，原来的逻辑更复杂
        if not hasattr(self, '__is_ivp_bvp'):
            self.__is_ivp_bvp = "BVP"  # 默认返回BVP
        return self.__is_ivp_bvp
    
    @_is_ivp_bvp.setter
    def _is_ivp_bvp(self, value):
        self.__is_ivp_bvp = value
