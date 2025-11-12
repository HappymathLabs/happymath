"""
重构后的DE基类 - 简化为核心功能
将复杂的表达式分析和符号管理功能委托给专门的组件
"""

from math import e
from typing import Union, List, Literal, Dict, Any
import sympy
import re
from sympy import classify_ode, classify_pde, dsolve, solve, lambdify, collect, diff, sympify, Basic
from sympy.solvers.ode.ode import solve_ics, _extract_funcs, constant_renumber, classify_sysode
from sympy.solvers.deutils import _preprocess, ode_order, _desolve
from sympy.solvers.ode.systems import _preprocess_eqs
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.add import Add
from sympy.core.power import Pow
from sympy.utilities.iterables import iterable
import logging

from .utils import check_symbol_type, sympy_assumptions, forced_trans_type
from ..diffeq_expr import analyze_expression
from .de_exceptions import (
    InvalidExpressionError
)

class DEBase(Expr):
    """
    重构后的微分方程基类
    委托复杂功能给专门的组件，专注于核心功能
    """
    
    def __new__(cls, sympy_obj: Union[Expr, iterable], value_range: str = "real", **kwargs):
        """
        继承自sympy.core.expr.Expr并处理表达式的强制转换
        
        Args:
            sympy_obj: 微分方程表达式或表达式列表
            value_range: 变量取值范围
            **kwargs: 其他参数
        """
        # 确保输入的取值范围合法
        if value_range not in sympy_assumptions:
            raise ValueError(f"{value_range} 不是有效的取值范围")

        # 整理输入的表达式
        if iterable(sympy_obj) and len(sympy_obj) == 1:
            sympy_obj = [item for item in sympy_obj][0]

        # 得到表达式(组)所有符号变量
        if iterable(sympy_obj):
            symbols = set.union(*(eq.free_symbols for eq in sympy_obj))
        else:
            symbols = sympy_obj.free_symbols

        # 检查取值范围
        sign_convert = False
        for symbol in symbols:
            if not check_symbol_type(symbol) == value_range:
                logging.warning(f"{symbol} 不是 {value_range} 变量，类型将被强制转换")
                sign_convert = True

        # 如果转换标志为真，则强制转换
        if sign_convert:
            converted_expr, convert_dict = cls._force_convert(symbols, sympy_obj, value_range)
            instance = super().__new__(cls, converted_expr)
            instance._sympy_obj = converted_expr
            instance._converted = True
            instance.convert_dict = convert_dict
            return instance
        else:
            instance = super().__new__(cls, sympy_obj)
            instance._sympy_obj = sympy_obj
            instance._converted = False
            instance.convert_dict = {}
            return instance

    @staticmethod
    def _force_convert(symbols, sympy_obj, value_range):
        """强制转换表达式中的符号为目标取值范围类型"""
        convert_dict = {}
        if iterable(sympy_obj):
            check_list = []
            for obj in sympy_obj:
                for symbol in symbols:
                    if not check_symbol_type(symbol) == value_range:
                        symbol_subs = forced_trans_type(symbol, value_range)
                        obj = obj.subs({symbol: symbol_subs})
                        convert_dict[symbol] = symbol_subs
                check_list.append(obj)
            return check_list, convert_dict
        else:
            for symbol in symbols:
                if not check_symbol_type(symbol) == value_range:
                    symbol_subs = forced_trans_type(symbol, value_range)
                    sympy_obj = sympy_obj.subs({symbol: symbol_subs})
                    convert_dict[symbol] = symbol_subs
            return sympy_obj, convert_dict

    def __init__(self, sympy_obj: Union[sympy.Expr, iterable], value_range: str = "real"):
        """
        初始化DEBase类
        
        Args:
            sympy_obj: 微分方程表达式或表达式列表
            value_range: 变量取值范围
        """
        # 验证取值范围
        if value_range not in sympy_assumptions:
            raise ValueError(f"{value_range} 不是有效的取值范围")
        else:
            self._value_range = value_range
        
        # 设置日志记录器
        self.logger = logging.getLogger(__name__)

    # 重写sympy.Expr的free_symbols，支持微分方程组
    @property
    def free_symbols(self) -> set:
        """
        返回表达式中的所有自由符号
        
        Returns:
            自由符号集合
        """
        if iterable(self._sympy_obj):
            return set.union(*(eq.free_symbols for eq in self._sympy_obj))
        else:
            return self._sympy_obj.free_symbols

    @property
    def expr(self) -> Union[Expr, List[Expr]]:
        """返回表达式对象"""
        return self._sympy_obj

    @property
    def is_ode(self) -> bool:
        """
        判断当前表达式是否为ODE
        委托给表达式分析器
        
        Returns:
            是否为ODE
        """
        try:
            analysis = analyze_expression(self._sympy_obj)
            et = analysis.get('expression_type', 'unknown')
            return et == 'ODE'
        except Exception:
            return False

    @property
    def is_pde(self) -> bool:
        """
        判断当前表达式是否为PDE
        委托给表达式分析器
        
        Returns:
            是否为PDE
        """
        try:
            analysis = analyze_expression(self._sympy_obj)
            et = analysis.get('expression_type', 'unknown')
            return et == 'PDE'
        except Exception:
            return False

    @property
    def is_linear(self) -> bool:
        """
        判断当前微分方程是否为线性
        委托给表达式分析器
        
        Returns:
            是否为线性
        """
        try:
            analysis = analyze_expression(self._sympy_obj)
            return analysis.get('is_linear', False)
        except Exception as e:
            self.logger.warning(f"线性检查失败: {e}")
            return False

    @property
    def free_funcs(self) -> set:
        """
        输出类中微分方程的所有函数变量集合
        委托给符号管理器
        
        Returns:
            函数变量集合
        """
        try:
            analysis = analyze_expression(self._sympy_obj)
            return set(analysis.get('core_functions', []))
        except Exception:
            return set()

    @property
    def free_consts(self) -> set:
        """
        输出类中微分方程的所有常数变量集合
        委托给符号管理器
        
        Returns:
            常数变量集合
        """
        try:
            analysis = analyze_expression(self._sympy_obj)
            return set(analysis.get('free_constants', []))
        except Exception:
            return set()

    @property
    def core_func(self) -> list:
        """
        输出微分方程的核心函数
        委托给符号管理器
        
        Returns:
            核心函数列表
        """
        try:
            analysis = analyze_expression(self._sympy_obj)
            return analysis.get('core_functions', [])
        except Exception:
            return []

    @property
    def core_symbol(self) -> list:
        """
        输出微分方程的核心变量
        委托给符号管理器
        
        Returns:
            核心符号列表
        """
        try:
            analysis = analyze_expression(self._sympy_obj)
            return analysis.get('core_symbols', [])
        except Exception:
            return []

    @property
    def core_func_symbol(self) -> dict:
        """
        输出核心函数与符号的映射字典
        委托给符号管理器
        
        Returns:
            函数与符号的映射字典
        """
        try:
            analysis = analyze_expression(self._sympy_obj)
            return analysis.get('core_func_symbol_mapping', {})
        except Exception:
            return {}
    
    @property
    def de_order(self) -> dict:
        """
        输出所有导数项与其阶数的映射字典
        委托给符号管理器
        
        Returns:
            导数项与阶数的映射字典
        """
        try:
            analysis = analyze_expression(self._sympy_obj)
            return analysis.get('derivative_orders', {})
        except Exception:
            return {}

    @property
    def order(self) -> int:
        """
        输出微分方程的最大阶数
        委托给符号管理器
        
        Returns:
            最大阶数
        """
        try:
            analysis = analyze_expression(self._sympy_obj)
            order_val = analysis.get('expression_order', None)
            if order_val is None:
                raise ValueError("缺少阶数信息")
            return int(order_val)
        except Exception as e:
            self.logger.error(f"获取阶数失败: {e}")
            raise InvalidExpressionError(self._sympy_obj, f"无法确定阶数: {e}")

    def _eqs2exprs(self, eqs: Union[Expr, iterable]) -> list:
        """
        将微分方程组转换为微分表达式组
        委托给表达式转换器
        
        Args:
            eqs: 方程或方程组
            
        Returns:
            表达式列表
        """
        try:
            from diffeq_expr.utils import eqs2exprs
            return eqs2exprs(eqs)
        except Exception:
            # 回退：最小实现
            try:
                if isinstance(eqs, list):
                    return [sympify(fi.lhs - fi.rhs) if isinstance(fi, sympy.Equality) else sympify(fi) for fi in eqs]
                else:
                    return [sympify(eqs.lhs - eqs.rhs) if isinstance(eqs, sympy.Equality) else sympify(eqs)]
            except Exception:
                return [eqs]

    def _is_number(self, s: Any) -> bool:
        """
        判断是否为数字
        
        Args:
            s: 待判断的对象
            
        Returns:
            是否为数字
        """
        if not isinstance(s, str):
            s = str(s)
        try:
            float(s)
            return True
        except ValueError:
            pass
        
        try:
            int(s)
            return True
        except (TypeError, ValueError):
            pass
        
        return False

    def _rename_key(self, dictionary: dict, old_key: Any, new_key: Any) -> dict:
        """
        在不改变顺序的前提下重命名key值
        
        Args:
            dictionary: 字典
            old_key: 旧键
            new_key: 新键
            
        Returns:
            重命名后的字典
        """
        if old_key in dictionary:
            # 保持顺序的字典重命名
            items = list(dictionary.items())
            new_items = []
            for key, value in items:
                if key == old_key:
                    new_items.append((new_key, value))
                else:
                    new_items.append((key, value))
            return dict(new_items)
        return dictionary

    # 为了向后兼容性，保留一些辅助分离方法的简化版本
    def _split_func_vars(self, func: Expr):
        """分离出函数中的函数名与变量符号"""
        func_name = func.func
        var_name = func.args
        return func_name, var_name

    def _split_der_funcs_vars(self, der_expr: Expr):
        """分离出导数表达式中的函数名与变量符号"""
        der_func = _extract_funcs([der_expr])
        func_name = []
        var_name = []
        
        for func in der_func:
            func_name.append(func.func)
            var_name.append(func.args)
        
        return der_func, func_name, var_name

    def _split_subs_funcs_vars(self, subs_expr):
        """分离出subs表达式中的函数名与变量符号"""
        func_name = subs_expr.expr
        var_name = subs_expr.variables
        cond_value = subs_expr.point
        
        return func_name, var_name, cond_value

    def _split_expr_meta(self, expr, mode_list=[Mul, Add, Pow]):
        """
        根据运算方式将表达式分离成元表达式
        委托给表达式转换器（保持接口兼容性）
        
        Args:
            expr: 表达式
            mode_list: 分离模式列表
            
        Returns:
            元表达式列表
        """
        try:
            from ..diffeq_expr.utils import split_expression_meta
            return split_expression_meta(expr, mode_list)
        except Exception as e:
            self.logger.warning(f"表达式分离失败: {e}")
            return [expr]

    # 验证方法
    def validate_expression(self) -> bool:
        """
        验证表达式的有效性
        
        Returns:
            是否有效
        """
        try:
            # 使用 ExprParser 的统一验证入口，兼容 ODE/PDE
            from ..diffeq_expr import validate_expression as _validate_expression
            result = _validate_expression(self._sympy_obj)
            return bool(result.get('is_valid', False))
        except Exception as e:
            self.logger.error(f"表达式验证失败: {e}")
            return False

    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        获取表达式分析摘要
        
        Returns:
            分析结果摘要
        """
        try:
            analysis = self._expression_analyzer.analyze_expression(self._sympy_obj)
            return {
                'type': analysis.get('type', 'unknown'),
                'is_ode': analysis.get('is_ode', False),
                'is_linear': analysis.get('is_linear', False),
                'order': analysis.get('order', 0),
                'num_functions': len(analysis.get('functions', [])),
                'num_variables': len(analysis.get('variables', [])),
                'has_derivatives': len(analysis.get('derivatives', [])) > 0
            }
        except Exception as e:
            self.logger.error(f"获取分析摘要失败: {e}")
            return {'error': str(e)}


