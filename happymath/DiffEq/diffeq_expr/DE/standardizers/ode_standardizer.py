"""
ODE标准化器
从ODEModule中迁移的标准化逻辑
"""

from typing import List, Dict, Union
import sympy
from sympy import solve
from sympy.utilities.iterables import iterable

from ...base.abstract_standardizer import AbstractStandardizer


class ODEStandardizer(AbstractStandardizer):
    """ODE标准化器"""
    
    def __init__(self, analyzer_result, symbol_manager_result):
        """
        初始化ODE标准化器
        
        Args:
            analyzer_result: ODE分析器结果对象
            symbol_manager_result: ODE符号管理器结果对象
        """
        super().__init__(analyzer_result, symbol_manager_result)
        self._is_system = (hasattr(analyzer_result, 'is_system_ode') and 
                          analyzer_result.is_system_ode())

    def standardize(self):
        """
        执行标准化处理
        统一调度：单一/高阶/方程组分支
        
        Returns:
            标准化器对象本身
        """
        try:
            if self._is_system:
                self._handle_system_ode()
            else:
                self._handle_single_ode()
                
            return self
            
        except Exception as e:
            raise ValueError(f"ODE标准化失败: {e}")

    def _handle_system_ode(self):
        """处理ODE方程组的标准化（不依赖 classify_sysode）"""
        from sympy.solvers.ode.systems import _preprocess_eqs

        # 预处理等式
        eqs = _preprocess_eqs(self.analyzer_result.sympy_obj)

        # 提取核心自变量与未知函数
        core_symbols = self.analyzer_result.core_symbols
        if not core_symbols:
            raise ValueError("未找到核心符号")
        core_symbol = core_symbols[0]
        core_funcs = list(self.analyzer_result.core_functions)
        if not core_funcs:
            raise ValueError("未找到未知函数")

        # 统计每个未知函数的最高阶（扫描 Derivative 原子）
        der_terms_dict = {}
        for eq in eqs:
            expr = eq.lhs - eq.rhs if isinstance(eq, sympy.Equality) else eq
            for der in expr.atoms(sympy.Derivative):
                try:
                    base = der.args[0]
                    order = len(der.variables)
                except Exception:
                    continue
                if base in core_funcs:
                    der_terms_dict[base] = max(der_terms_dict.get(base, 0), order)

        if not der_terms_dict:
            raise ValueError("未能识别到导数项，无法标准化系统")

        # 如果所有最高阶均为 1，则按一阶系统处理：逐方程分离一阶导数
        if all(v == 1 for v in der_terms_dict.values()):
            candidates = [func.diff(core_symbol, 1) for func in der_terms_dict.keys()]
            stand_ode_list = []
            for eq in eqs:
                if isinstance(eq, sympy.Equality) and isinstance(eq.lhs, sympy.Derivative):
                    # 已经是显式一阶导数形式
                    stand_ode_list.append(eq)
                    continue
                expr = eq.lhs - eq.rhs if isinstance(eq, sympy.Equality) else eq
                separated = None
                for cand in candidates:
                    try:
                        sol = solve(sympy.Eq(expr, 0), cand)
                        if sol:
                            separated = sympy.Eq(cand, sol[0])
                            break
                    except Exception:
                        pass
                if separated is None:
                    raise ValueError("无法将系统分离为显式一阶导数形式")
                stand_ode_list.append(separated)
            self.standardized_expressions = stand_ode_list
            self.substitution_dict = {}
            return

        # 否则按多阶系统处理：为低阶导数创建替代符号，分离最高阶导数
        Y_symbols = self.symbol_manager_result.get_generated_symbols()
        self._handle_multi_order_system(der_terms_dict, Y_symbols, core_symbol)
    
    def _is_first_order_single_system(self, der_terms_dict, eqs) -> bool:
        """判断是否为一阶单一方程组"""
        highest_order = sum([i for i in der_terms_dict.values()])
        return highest_order == 1 and len(eqs) == 1
    
    def _handle_first_order_single_system(self, eqs, der_terms_dict, Y_symbols):
        """处理一阶单一ODE系统"""
        core_symbols = self.analyzer_result.core_symbols
        if not core_symbols:
            raise ValueError("未找到核心符号")
        core_symbol = core_symbols[0]
        
        obj = self.analyzer_result.sympy_obj[0]
        if isinstance(obj, sympy.Equality):
            obj = obj.lhs - obj.rhs
            
        core_func = list(der_terms_dict.keys())[0]
        separate_diff = core_func.diff(core_symbol, 1)
        
        try:
            separate_expr = solve(sympy.Eq(obj, 0), separate_diff)[0]
            eq_subs = sympy.Eq(separate_diff, separate_expr)
            self.standardized_expressions = [eq_subs]
        except (IndexError, Exception):
            raise ValueError(f"无法分离导数项 {separate_diff}")
        
        # 设置其他属性
        self.substitution_dict = {}
    
    def _handle_multi_order_system(self, der_terms_dict, Y_symbols, core_symbol):
        """处理多阶ODE系统"""
        subs_dict = {}
        stand_ode_list = []
        cnt_idx = 0
        
        # 根据核心函数的阶数创建替代字典和替代等式
        for core_func in der_terms_dict.keys():
            for index in range(0, der_terms_dict[core_func] - 1):
                if index + cnt_idx < len(Y_symbols):
                    eq_subs = sympy.Eq(
                        core_func.diff(core_symbol, index + 1), 
                        Y_symbols[index + cnt_idx]
                    ).subs(subs_dict)
                    subs_dict[core_func.diff(core_symbol, index + 1)] = Y_symbols[index + cnt_idx]
                    stand_ode_list.append(eq_subs)
            cnt_idx += (der_terms_dict[core_func] - 1)
            
        # 使用替代字典替换表达式，并单独分离出阶数最高项表达式
        for index, obj in enumerate(self.analyzer_result.sympy_obj):
            if isinstance(obj, sympy.Equality):
                obj = obj.lhs - obj.rhs
                
            for core_func in der_terms_dict.keys():
                separate_diff = core_func.diff(core_symbol, der_terms_dict[core_func])
                
                try:
                    separate_expr = solve(sympy.Eq(obj, 0), separate_diff)
                    
                    if len(separate_expr) == 0:
                        continue
                    else:
                        eq_separate = sympy.Eq(separate_diff, separate_expr[0]).subs(subs_dict)
                        stand_ode_list.append(eq_separate)
                except (IndexError, Exception):
                    continue
                    
        self.standardized_expressions = stand_ode_list
        self.substitution_dict = subs_dict

    def _handle_single_ode(self):
        """处理单个ODE"""
        obj = self.analyzer_result.sympy_obj
        if isinstance(obj, sympy.Equality):
            obj = obj.lhs - obj.rhs
            
        order_num = self.analyzer_result.expression_order
        core_functions = self.analyzer_result.core_functions
        core_symbols = self.analyzer_result.core_symbols
        
        if not core_functions or not core_symbols:
            raise ValueError("未找到核心函数或符号")
            
        core_func = core_functions[0]
        core_symbol = core_symbols[0]
        
        Y_symbols = self.symbol_manager_result.get_generated_symbols()
        
        subs_dict = {}
        stand_ode_list = []
        bool_first = True
        
        for index in range(0, order_num):
            if index == order_num - 1:
                if bool_first:  # 一阶ODE
                    Y_symbol = core_func.diff(core_symbol, index + 1)
                    try:
                        separate_expr = solve(sympy.Eq(obj.subs(subs_dict), 0), Y_symbol)[0]
                        eq = sympy.Eq(Y_symbol, separate_expr)
                        stand_ode_list.append(eq)
                    except (IndexError, Exception):
                        raise ValueError(f"无法分离导数项 {Y_symbol}")
                    break
                else:  # 高阶ODE
                    if index - 1 < len(Y_symbols):
                        Y_tmp = Y_symbols[index - 1]
                        Y_symbol = sympy.Symbol(f"{Y_tmp}")
                        separate_diff = sympy.Derivative(Y_symbol, core_symbol)
                        
                        try:
                            separate_expr = solve(sympy.Eq(obj.subs(subs_dict), 0), separate_diff)[0]
                            eq = sympy.Eq(separate_diff, separate_expr)
                            stand_ode_list.append(eq)
                        except (IndexError, Exception):
                            raise ValueError(f"无法分离导数项 {separate_diff}")
                    break
                    
            # 通过递推将导数项进行替换
            bool_first = False
            if index < len(Y_symbols):
                eq = sympy.Eq(core_func.diff(core_symbol, index + 1), Y_symbols[index]).subs(subs_dict)
                subs_dict[core_func.diff(core_symbol, index + 1)] = Y_symbols[index]
                stand_ode_list.append(eq)
                
        self.standardized_expressions = stand_ode_list
        self.substitution_dict = subs_dict

    def build_substitution_system(self) -> List[sympy.Eq]:
        """
        构建替代方程组
        
        Returns:
            替代方程组列表
        """
        return [eq for eq in self.standardized_expressions 
                if any(sym in eq.rhs.atoms() for sym in self.symbol_manager_result.get_generated_symbols())]

    def separate_highest_order_terms(self) -> List[sympy.Eq]:
        """
        分离最高阶导数项
        
        Returns:
            分离后的方程列表
        """
        highest_order_eqs = []
        
        for eq in self.standardized_expressions:
            # 检查是否包含最高阶导数
            if isinstance(eq.lhs, sympy.Derivative):
                derivative_order = len(eq.lhs.args) - 1
                if derivative_order == self.analyzer_result.expression_order:
                    highest_order_eqs.append(eq)
        
        return highest_order_eqs

    def create_standard_form(self) -> List[sympy.Eq]:
        """
        创建标准形式
        
        Returns:
            标准形式方程列表
        """
        return self.standardized_expressions

    def get_substitution_dict(self) -> Dict:
        """
        获取替代字典
        
        Returns:
            替代字典
        """
        return self.substitution_dict.copy()
    
    def get_undetermined_terms(self) -> List:
        """
        获取未定项列表
        
        Returns:
            未定项列表
        """
        undetermined_terms = []
        
        # 添加替代字典中的导数项
        for und_der in self.substitution_dict.keys():
            undetermined_terms.append(und_der)
            
        # 添加核心函数
        if hasattr(self.analyzer_result, 'core_functions'):
            for und_func in self.analyzer_result.core_functions:
                undetermined_terms.append(und_func)
                
        return undetermined_terms
