"""
ODE分析器
从ODEModule和现有expression_analyzer中迁移的ODE分析功能
"""

from typing import List, Dict, Set, Union
import sympy
from sympy import classify_ode
from sympy.solvers.ode.ode import classify_sysode
from sympy.solvers.ode.systems import _preprocess_eqs
from sympy.solvers.ode.ode import _extract_funcs
from sympy.utilities.iterables import iterable

from ...base.abstract_analyzer import AbstractExpressionAnalyzer
from ...utils import eqs2exprs


class ODEAnalyzer(AbstractExpressionAnalyzer):
    """ODE表达式分析器"""
    
    def __init__(self, sympy_obj: Union[sympy.Expr, list], value_range: str = 'real'):
        """
        初始化ODE分析器
        
        Args:
            sympy_obj: ODE表达式或表达式列表
            value_range: 变量取值范围
        """
        super().__init__(sympy_obj, value_range)
        self._ode_classification = None
        self._is_system = iterable(sympy_obj) and len(sympy_obj) > 1

    def is_valid_expression(self) -> bool:
        """
        验证是否为有效的ODE表达式
        统一判断单方程与方程组
        
        Returns:
            是否为有效的ODE表达式
        """
        if self.get_cached_value('is_valid') is not None:
            return self.get_cached_value('is_valid')
        
        try:
            if self._is_system:
                # 对于常微分方程组，核心变量只能是一个
                core_symbols = self.core_symbols
                if len(core_symbols) > 1:
                    self.set_cached_value('is_valid', False)
                    return False
                    
                # 预处理方程组后再分类
                eqs = _preprocess_eqs(self.sympy_obj)
                self._ode_classification = classify_sysode(eqs)
            else:
                # 单一方程处理
                eq = self.sympy_obj if isinstance(self.sympy_obj, sympy.Equality) else sympy.Eq(self.sympy_obj, 0)
                self._ode_classification = classify_ode(eq)
            
            self.set_cached_value('is_valid', True)
            return True
            
        except Exception as e:
            # 回退：不依赖分类器的最小ODE有效性检查
            try:
                eq_list = eqs2exprs(self.sympy_obj)
                derivatives = []
                for eq in eq_list:
                    derivatives.extend(list(eq.atoms(sympy.Derivative)))
                # 必须至少包含一个导数
                if not derivatives:
                    self.set_cached_value('is_valid', False)
                    return False
                # 所有导数的自变量应当一致（单一自变量）
                indep_vars = set()
                for der in derivatives:
                    if der.variables:
                        # 变量可能以 (x, 1) 的形式出现
                        for v in der.variables:
                            indep_vars.add(v)
                if len(indep_vars) > 1:
                    self.set_cached_value('is_valid', False)
                    return False
                self.set_cached_value('is_valid', True)
                return True
            except Exception:
                self.set_cached_value('is_valid', False)
                return False

    @property
    def expression_type(self) -> str:
        """表达式类型"""
        return 'ODE'

    @property
    def is_linear(self) -> bool:
        """
        判断ODE是否为线性
        基于现有的线性检查逻辑
        """
        if self.get_cached_value('is_linear') is not None:
            return self.get_cached_value('is_linear')
        
        try:
            # 获取表达式中的函数调用（如 y(t)、y1(x)）
            if self._is_system:
                # 对于方程组，检查每个方程的线性性
                for expr in self.sympy_obj:
                    if not self._check_single_expression_linearity(expr):
                        self.set_cached_value('is_linear', False)
                        return False
                self.set_cached_value('is_linear', True)
                return True
            else:
                result = self._check_single_expression_linearity(self.sympy_obj)
                self.set_cached_value('is_linear', result)
                return result
                
        except Exception:
            self.set_cached_value('is_linear', False)
            return False
    
    def _check_single_expression_linearity(self, expr: sympy.Expr) -> bool:
        """检查单个表达式的线性性（稳健版）"""
        from sympy.core.function import AppliedUndef
        from sympy import Derivative, Poly, Dummy, Function, Pow, Mul
        from sympy.polys.polyerrors import PolynomialError

        # 统一到等式左减右的形式，并尽可能展开导数（乘积法则等）
        if isinstance(expr, sympy.Equality):
            expr = expr.lhs - expr.rhs

        try:
            expr = sympy.expand(expr.doit())
        except Exception:
            expr = sympy.expand(expr)

        # 识别未知解函数：凡是在整个问题中出现在导数里的函数调用，均视为未知解函数
        # 这能有效排除系数函数（如 p(t)）
        try:
            eq_list = eqs2exprs(self.sympy_obj)
        except Exception:
            eq_list = [self.sympy_obj] if not isinstance(self.sympy_obj, list) else list(self.sympy_obj)

        unknown_calls_all = set()
        for e in eq_list:
            for d in e.atoms(Derivative):
                if d.args:
                    base = d.args[0]
                    if isinstance(base, AppliedUndef):
                        unknown_calls_all.add(base)

        # 提取该表达式中出现的未知函数和其导数
        present_funcs = set(expr.atoms(AppliedUndef)) & unknown_calls_all
        present_derivs = set(d for d in expr.atoms(Derivative) if d.args and d.args[0] in unknown_calls_all)

        unknown_terms = list(present_funcs | present_derivs)

        # 若不存在未知项（只有自变量/常数/已知函数），则视为线性
        if not unknown_terms:
            return True

        # 将未知函数及其各阶导数替换为代数占位符，检查关于这些占位符是否至多一次项
        replacements = {}
        dummies = []
        for idx, term in enumerate(unknown_terms):
            dm = Dummy(f"_U{idx}")
            replacements[term] = dm
            dummies.append(dm)

        substituted_expr = sympy.expand(expr.xreplace(replacements))

        # 非线性包装：任何包含占位符的函数调用都是非线性的，例如 sin(U)、exp(U)
        for fcall in substituted_expr.atoms(Function):
            # 如果函数参数中包含任何占位符，则非线性
            if any(dm in fcall.free_symbols for dm in dummies):
                return False

        # 非线性幂次：占位符位于幂的底数且指数不是1
        for pw in substituted_expr.atoms(Pow):
            if any(dm in pw.base.free_symbols for dm in dummies):
                try:
                    if not (pw.exp.is_Integer and int(pw.exp) == 1):
                        return False
                except Exception:
                    return False

        # 非线性交叉项：任一乘积项中占位符出现超过一次（包括不同占位符）
        for mul in substituted_expr.atoms(Mul):
            count = 0
            for dm in dummies:
                try:
                    # 统计该占位符在乘积中的幂次（若未出现则为0）
                    deg = mul.as_powers_dict().get(dm, 0)
                except Exception:
                    deg = 0
                if deg:
                    # 只要幂次>=1视为出现一次
                    if deg != 1:
                        # 幂次>1 直接非线性
                        return False
                    count += 1
                if count > 1:
                    return False

        # 不能出现对未知项的非多项式依赖（如 sin(U), exp(U), Abs(U), U1*U2, U**2 等）
        # 用多项式形式做最后的兜底（拓展域，允许一般表达式作为系数）
        try:
            poly = Poly(substituted_expr, *dummies, domain='EX')
            if poly.total_degree() > 1:
                return False
            for sym in dummies:
                deg = poly.degree(sym)
                if deg is not None and deg > 1:
                    return False
        except PolynomialError:
            # 无法表示为关于占位符的多项式，视为非线性
            return False
        except Exception:
            return False

        return True
    
    def _get_expression_order(self, expr: sympy.Expr) -> int:
        """获取表达式的阶数"""
        max_order = 0
        for deriv in expr.atoms(sympy.Derivative):
            order = len(deriv.args) - 1  # 去除函数本身
            if order > max_order:
                max_order = order
        return max_order

    @property
    def core_functions(self) -> List[sympy.Function]:
        """
        提取核心函数列表
        使用现有的_extract_funcs逻辑
        """
        if self.get_cached_value('core_functions') is not None:
            return self.get_cached_value('core_functions')
        
        eq_list = eqs2exprs(self.sympy_obj)
        funcs_list = _extract_funcs(eq_list)
        self.set_cached_value('core_functions', funcs_list)
        return funcs_list

    @property
    def core_symbols(self) -> List[sympy.Symbol]:
        """
        提取核心符号列表
        基于核心函数与符号映射提取
        """
        if self.get_cached_value('core_symbols') is not None:
            return self.get_cached_value('core_symbols')
        
        func_symbol_dict = self.core_func_symbol_mapping
        core_symbol = []
        
        for value in func_symbol_dict.values():
            if isinstance(value, tuple):
                core_symbol.extend(value)
            else:
                core_symbol.append(value)
        
        # 去除重复并返回
        result = list(set(core_symbol))
        self.set_cached_value('core_symbols', result)
        return result

    @property
    def derivative_orders(self) -> Dict[sympy.Derivative, int]:
        """
        提取导数项与阶数的映射
        """
        if self.get_cached_value('derivative_orders') is not None:
            return self.get_cached_value('derivative_orders')
        
        eq_list = eqs2exprs(self.sympy_obj)
        derivative_dict = {}
        
        # 提取所有导数项
        from ...utils import flatten_list
        der_list = [list(eq.atoms(sympy.Derivative)) for eq in eq_list]
        derivatives = flatten_list(der_list)
        
        for der in derivatives:
            order = len(der.variables)
            derivative_dict[der] = order
        
        self.set_cached_value('derivative_orders', derivative_dict)
        return derivative_dict

    @property
    def free_constants(self) -> Set[sympy.Symbol]:
        """
        提取自由常数集合
        """
        if self.get_cached_value('free_constants') is not None:
            return self.get_cached_value('free_constants')
        
        core_symbols = self.core_symbols
        consts_set = set()
        
        def consts_in_expr(expression):
            for symbol in expression.free_symbols:
                if symbol not in core_symbols:
                    consts_set.add(symbol)
        
        if iterable(self.sympy_obj):
            for obj in self.sympy_obj:
                consts_in_expr(obj)
        else:
            consts_in_expr(self.sympy_obj)
        
        self.set_cached_value('free_constants', consts_set)
        return consts_set

    @property
    def expression_order(self) -> int:
        """
        获取表达式最大阶数
        """
        if self.get_cached_value('expression_order') is not None:
            return self.get_cached_value('expression_order')
        
        der_terms_dict = self.derivative_orders
        
        if not der_terms_dict:
            order_value = 0
        else:
            order_value = max(der_terms_dict.values())
        
        if order_value == 0:
            raise ValueError("微分方程的阶数为零，请检查输入")
        
        self.set_cached_value('expression_order', order_value)
        return order_value

    @property
    def core_func_symbol_mapping(self) -> Dict:
        """
        获取核心函数与符号的映射关系
        """
        if self.get_cached_value('core_func_symbol_mapping') is not None:
            return self.get_cached_value('core_func_symbol_mapping')
        
        eq_list = eqs2exprs(self.sympy_obj)
        from ...utils import flatten_list
        der_list = [list(eq.atoms(sympy.Derivative)) for eq in eq_list]
        flattened_list = flatten_list(der_list)
        
        func_symbol_dict = {}
        
        for flatten_item in flattened_list:
            # 检查混合偏导数（ODE中不应该出现）
            if (len(flatten_item.variables) > 1 and 
                flatten_item.variables[0] != flatten_item.variables[1]):
                raise ValueError("ODE中不应包含混合偏导数")
            
            func_item = _extract_funcs([flatten_item])[0]
            if func_item not in func_symbol_dict:
                func_symbol_dict[func_item] = []
            func_symbol_dict[func_item].append(flatten_item.variables[0])
        
        # 整理字典
        for func, symbols in func_symbol_dict.items():
            if len(symbols) == 1:
                func_symbol_dict[func] = symbols[0]
            else:
                func_symbol_dict[func] = tuple(set(symbols))
        
        self.set_cached_value('core_func_symbol_mapping', func_symbol_dict)
        return func_symbol_dict
    
    def get_ode_classification(self):
        """获取ODE分类结果"""
        if self._ode_classification is None:
            self.is_valid_expression()  # 触发分类
        return self._ode_classification
    
    def is_system_ode(self) -> bool:
        """判断是否为ODE方程组"""
        return self._is_system