"""
PDE分析器
从PDEModule中迁移的PDE分析功能
"""

from typing import List, Dict, Set, Union
import sympy
import warnings
from sympy.utilities.iterables import iterable
from sympy.polys.polyerrors import PolynomialError

from ...base.abstract_analyzer import AbstractExpressionAnalyzer
from ...utils import eqs2exprs, flatten_list


class PDEAnalyzer(AbstractExpressionAnalyzer):
    """PDE表达式分析器"""
    
    def __init__(self, sympy_obj: Union[sympy.Expr, list], value_range: str = 'real', 
                 spatial_var_order: List[str] = None):
        """
        初始化PDE分析器
        
        Args:
            sympy_obj: PDE表达式或表达式列表
            value_range: 变量取值范围
            spatial_var_order: 空间变量顺序，默认为['x', 'y']
        """
        super().__init__(sympy_obj, value_range)
        
        if spatial_var_order is None:
            spatial_var_order = ['x', 'y']
        self.spatial_var_order = list(spatial_var_order)
        
        self.time_var = None
        self.spatial_vars: List[sympy.Symbol] = []
        self._var_classification: Dict = {}
        
        # 执行变量分类
        self._classify_variables()

    def is_valid_expression(self) -> bool:
        """
        验证是否为有效的PDE表达式
        检查是否包含多自变量未知函数及其偏导数（不强制包含时间变量t）
        
        Returns:
            是否为有效的PDE表达式
        """
        if self.get_cached_value('is_valid') is not None:
            return self.get_cached_value('is_valid')
        
        try:
            # 统一提取等式为左-右形式
            eq_list = eqs2exprs(self.sympy_obj)
            
            # 1) 必须包含导数项
            all_derivatives = []
            for eq in eq_list:
                all_derivatives.extend(list(eq.atoms(sympy.Derivative)))
            if not all_derivatives:
                self.set_cached_value('is_valid', False)
                return False
            
            # 2) 必须存在未知多元函数（自变量数量≥2）
            core_funcs = self.core_functions
            if not core_funcs:
                self.set_cached_value('is_valid', False)
                return False
            multi_var_funcs = [f for f in core_funcs if hasattr(f, 'args') and len(f.args) >= 2]
            if not multi_var_funcs:
                self.set_cached_value('is_valid', False)
                return False
            
            # 3) 至少存在一个对未知多元函数的偏导项
            derivs_of_unknown = [d for d in all_derivatives if d.args and d.args[0] in multi_var_funcs]
            if not derivs_of_unknown:
                self.set_cached_value('is_valid', False)
                return False
            
            # 4) 空间维度限制（当前仅支持≤2个非t自变量）
            spatial_vars_set = set()
            for f in multi_var_funcs:
                for arg in getattr(f, 'args', []):
                    if isinstance(arg, sympy.Symbol) and str(arg) != 't':
                        spatial_vars_set.add(arg)
            if len(spatial_vars_set) == 0:
                # 没有任何非t自变量，不是PDE
                self.set_cached_value('is_valid', False)
                return False
            if len(spatial_vars_set) > 2:
                # 超出当前支持范围
                self.set_cached_value('is_valid', False)
                return False
            
            # 5) 混合偏导数仅报警告
            if self.check_mixed_derivatives():
                warnings.warn("Mixed partial derivatives detected, which may not be fully supported at this time.")
            
            self.set_cached_value('is_valid', True)
            return True
            
        except Exception as e:
            self.set_cached_value('is_valid', False)
            return False

    def _check_core_symbol(self, symbol_str: str) -> bool:
        """检查是否包含指定的核心符号"""
        core_symbols = self.core_symbols
        for symbol in core_symbols:
            if str(symbol) == symbol_str:
                return True
        return False

    @property
    def expression_type(self) -> str:
        """表达式类型"""
        return 'PDE'

    @property
    def is_linear(self) -> bool:
        """
        判断PDE是否为线性
        改进后的严格判定：
        - 将所有未知函数及其出现的偏导数替换为占位符（Dummy）
        - 占位符不能出现在任意函数的参数、分母或指数中
        - 以占位符为元尝试构造多项式；若失败或总次数>1，则非线性
        
        Returns:
            是否为线性PDE
        """
        if self.get_cached_value('is_linear') is not None:
            return self.get_cached_value('is_linear')
        
        try:
            # 统一表达式为列表形式并转换为 lhs - rhs
            eq_list = eqs2exprs(self.sympy_obj)

            # 识别未知解函数：凡是在导数中的函数调用均视为未知解函数（排除系数函数）
            from sympy.core.function import AppliedUndef
            unknown_calls_all: Set[sympy.Expr] = set()
            for eq in eq_list:
                for d in eq.atoms(sympy.Derivative):
                    if d.args:
                        base = d.args[0]
                        if isinstance(base, AppliedUndef):
                            unknown_calls_all.add(base)

            if not unknown_calls_all:
                self.set_cached_value('is_linear', False)
                return False

            unknown_funcs = list(unknown_calls_all)

            # 收集所有与未知函数相关的偏导数（按展开后的表达式更稳健）
            derivative_set: Set[sympy.Derivative] = set()
            for eq in eq_list:
                eq_expanded = sympy.expand(eq.doit())
                for der in eq_expanded.atoms(sympy.Derivative):
                    base = der.args[0] if der.args else None
                    if base in unknown_calls_all:
                        derivative_set.add(der)

            # 为未知函数及其偏导创建占位符
            placeholder_map: Dict[Union[sympy.Expr, sympy.Derivative], sympy.Symbol] = {}

            # 函数占位符，如 u(x,y) -> ζ_u
            for func in unknown_funcs:
                base_name = str(getattr(func, 'func', func))
                placeholder_map[func] = sympy.Dummy(f"__U_{base_name}")

            # 导数占位符，如 Derivative(u(x,y), x, y) -> ζ_u_dx_dy
            for der in derivative_set:
                base_name = str(getattr(der.args[0], 'func', der.args[0]))
                var_names = "_".join([str(v) for v in der.variables])
                placeholder_map[der] = sympy.Dummy(f"__U_{base_name}_d_{var_names}")

            placeholder_symbols: Set[sympy.Symbol] = set(placeholder_map.values())

            # 针对每个等式进行判定
            for raw_expr in eq_list:
                # 先展开导数与乘积，提升线性判定稳健性（如散度形式）
                expr = sympy.expand(raw_expr.doit())

                # 执行替换
                expr_replaced = expr.subs(placeholder_map, simultaneous=True)

                # 1) 占位符不能出现在任意函数的参数中（如 sin(ζ) 等）
                for fn in expr_replaced.atoms(sympy.Function):
                    # 若任一参数包含占位符，则非线性
                    for arg in getattr(fn, 'args', []):
                        if arg.free_symbols & placeholder_symbols:
                            self.set_cached_value('is_linear', False)
                            return False

                # 2) 占位符不能出现在分母中
                num, den = sympy.fraction(expr_replaced)
                if den.free_symbols & placeholder_symbols:
                    self.set_cached_value('is_linear', False)
                    return False

                # 3) 占位符不能出现在幂的指数中，且若为底数，仅允许指数为 1
                for pw in expr_replaced.atoms(sympy.Pow):
                    base_syms = pw.base.free_symbols & placeholder_symbols
                    exp_syms = pw.exp.free_symbols & placeholder_symbols
                    if exp_syms:
                        self.set_cached_value('is_linear', False)
                        return False
                    if base_syms:
                        # 仅允许 pow 的指数等于 1（否则如 ζ**2、sqrt(ζ) 等属于非线性）
                        try:
                            if not sympy.simplify(pw.exp - 1) == 0:
                                self.set_cached_value('is_linear', False)
                                return False
                        except Exception:
                            self.set_cached_value('is_linear', False)
                            return False

                # 4) 用多项式视角检查占位符的总次数 ≤ 1
                try:
                    poly = sympy.Poly(num, *sorted(placeholder_symbols, key=lambda s: s.name))
                except PolynomialError:
                    self.set_cached_value('is_linear', False)
                    return False

                if poly.total_degree() > 1:
                    self.set_cached_value('is_linear', False)
                    return False

            self.set_cached_value('is_linear', True)
            return True

        except Exception:
            self.set_cached_value('is_linear', False)
            return False

    @property
    def core_functions(self) -> List[sympy.Function]:
        """
        提取核心函数列表
        """
        if self.get_cached_value('core_functions') is not None:
            return self.get_cached_value('core_functions')
        
        from sympy.solvers.ode.ode import _extract_funcs
        eq_list = eqs2exprs(self.sympy_obj)
        funcs_list = _extract_funcs(eq_list)
        
        self.set_cached_value('core_functions', funcs_list)
        return funcs_list

    @property
    def core_symbols(self) -> List[sympy.Symbol]:
        """
        提取核心符号列表
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
        der_list = [list(eq.atoms(sympy.Derivative)) for eq in eq_list]
        derivatives = flatten_list(der_list)
        
        for der in derivatives:
            # 检查混合偏导数
            if (len(der.variables) > 1 and 
                der.variables[0] != der.variables[1]):
                warnings.warn(f"检测到混合偏导数: {der}")
            
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
        der_list = [list(eq.atoms(sympy.Derivative)) for eq in eq_list]
        flattened_list = flatten_list(der_list)
        
        func_symbol_dict = {}
        
        for flatten_item in flattened_list:
            from sympy.solvers.ode.ode import _extract_funcs
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

    def _classify_variables(self):
        """
        分类变量，识别时间变量和空间变量
        迁移自PDEModule变量分类逻辑
        """
        core_symbols = self.core_symbols
        
        # 定义一个空间变量字典，根据list顺序分别存储x,y,z,...这些空间变量
        self.spatial_var_list = []
        self.time_var = None
        
        for var in core_symbols:
            if str(var) != "t":
                if str(var) != "x" and str(var) != "y":
                    warnings.warn(f"{str(var)} 不是标准的空间变量，将根据获取顺序映射到x、y或z轴。"
                                f"您可以使用 'spatial_var_order' 来指定空间变量的顺序。")
                self.spatial_var_list.append(str(var))
            else:
                self.time_var = var
        
        # 根据spatial_var_order对self.spatial_var_list进行排序
        self.spatial_order_var_list = [var for var in self.spatial_var_order if var in self.spatial_var_list]
        if len(self.spatial_order_var_list) == len(self.spatial_var_list):
            self.spatial_var_list = self.spatial_order_var_list
        else:
            warnings.warn(f"表达式中的空间变量与指定顺序不匹配。将使用顺序 {self.spatial_var_list} 作为空间变量。")
        
        # 更新空间变量符号列表
        self.spatial_vars = [sympy.Symbol(var_str) for var_str in self.spatial_var_list]
        
        # 构建变量分类字典
        self._var_classification = {
            't': self.time_var,
            'spatial': self.spatial_vars,
            'spatial_names': self.spatial_var_list
        }

    def classify_variables(self):
        """
        变量分类接口
        
        Returns:
            变量分类字典
        """
        return {'t': self.time_var, 'xys': self.spatial_vars}

    def validate_spatial_dimensions(self) -> bool:
        """
        验证空间维度
        当前限制 ≤ 2 维
        
        Returns:
            是否满足维度限制
        """
        return len(self.spatial_var_list) <= 2

    def check_mixed_derivatives(self) -> bool:
        """
        检查是否含混合偏导数
        
        Returns:
            是否含混合偏导数
        """
        derivatives = self.derivative_orders
        
        for deriv in derivatives.keys():
            if (len(deriv.variables) > 1 and 
                deriv.variables[0] != deriv.variables[1]):
                return True
        
        return False
    
    def get_time_derivatives(self) -> Dict[sympy.Derivative, int]:
        """获取时间导数项"""
        time_derivatives = {}
        derivatives = self.derivative_orders
        
        for deriv, order in derivatives.items():
            if str(deriv.args[1][0]) == "t":  # 检查是否为时间导数
                time_derivatives[deriv] = order
        
        return time_derivatives
    
    def get_spatial_derivatives(self) -> Dict[sympy.Derivative, int]:
        """获取空间导数项"""
        spatial_derivatives = {}
        derivatives = self.derivative_orders
        
        for deriv, order in derivatives.items():
            if str(deriv.args[1][0]) != "t":  # 检查是否为空间导数
                spatial_derivatives[deriv] = order
        
        return spatial_derivatives
    
    def get_spatial_var_list(self) -> List[str]:
        """获取空间变量名称列表"""
        return self.spatial_var_list.copy()
    
    def get_time_var(self) -> sympy.Symbol:
        """获取时间变量"""
        return self.time_var