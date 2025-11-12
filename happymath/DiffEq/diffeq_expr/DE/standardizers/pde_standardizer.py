"""
PDE标准化器
从PDEModule中迁移的标准化逻辑
"""

from typing import List, Dict, Union
import sympy
from sympy import solve
from sympy.utilities.iterables import iterable

from ...base.abstract_standardizer import AbstractStandardizer
from ...utils import split_expression_meta, split_derivative_components, split_func_vars


class PDEStandardizer(AbstractStandardizer):
    """PDE标准化器"""
    
    def __init__(self, analyzer_result, symbol_manager_result):
        """
        初始化PDE标准化器
        
        Args:
            analyzer_result: PDE分析器结果对象
            symbol_manager_result: PDE符号管理器结果对象
        """
        super().__init__(analyzer_result, symbol_manager_result)
        
        # PDE特有的属性
        self.time_de_order_dict = {}  # 时间导数项及其阶数
        self.space_de_order_dict = {}  # 空间导数项及其阶数
        self.sub_obj = []  # 替换导数项的函数组
        self.subs_dict = {}  # 替换导数项的字典
        self.solvable_format = {}  # 面向py-pde的可求解格式

    def standardize(self):
        """
        执行标准化处理
        构建时间导数替代、空间导数映射、标准表达式
        
        Returns:
            标准化器对象本身
        """
        try:
            # 初始化属性
            self.time_de_order_dict = {}
            self.space_de_order_dict = {}
            self.sub_obj = []
            self.subs_dict = {}
            
            # 处理表达式
            if not iterable(self.analyzer_result.sympy_obj):
                obj = [self.analyzer_result.sympy_obj]
            else:
                obj = self.analyzer_result.sympy_obj
            
            # 转换为标准形式
            expr_list = []
            for eqs_item in obj:
                if isinstance(eqs_item, sympy.Equality):
                    eqs_item = eqs_item.lhs - eqs_item.rhs
                expr_list.append(eqs_item)
            
            # 获取导数阶数信息
            de_order = self.analyzer_result.derivative_orders
            
            # 分类时间和空间导数
            self._classify_derivatives(de_order)
            
            # 计算替代符号数量并生成
            sub_obj_num = self._calculate_substitution_count()
            
            # 生成替代对象
            if sub_obj_num > 0:
                sub_obj_symbols = self.symbol_manager_result.generate_substitute_symbols(
                    sub_obj_num, mode="function"
                )
                self.sub_obj = sub_obj_symbols
            
            # 检查是否为一阶PDE
            if self._is_first_order_single_pde(expr_list):
                self._handle_first_order_pde(expr_list)
            else:
                self._handle_higher_order_pde(expr_list)
            
            return self
            
        except Exception as e:
            raise ValueError(f"PDE标准化失败: {e}")
    
    def _classify_derivatives(self, de_order):
        """分类时间和空间导数"""
        for de_item, order_value in de_order.items():
            if str(de_item.args[1][0]) == "t":
                self.time_de_order_dict[de_item] = order_value
            else:
                self.space_de_order_dict[de_item] = order_value
    
    def _calculate_substitution_count(self) -> int:
        """计算替代符号数量"""
        # 计算每个函数关于时间的最高阶数
        func_max_order = {}
        for de_item, order_value in self.time_de_order_dict.items():
            func = de_item.args[0]  # 获取被微分的函数
            func_max_order[func] = max(func_max_order.get(func, 0), order_value)
        
        # 计算替代符号数量为每个函数最高阶数-1的和
        sub_obj_num = 0
        for func, max_order in func_max_order.items():
            sub_obj_num += max(0, max_order - 1)
        
        return sub_obj_num
    
    def _is_first_order_single_pde(self, expr_list) -> bool:
        """判断是否为一阶单一PDE"""
        highest_order = sum([order for order in self.time_de_order_dict.values()])
        return highest_order == 1 and len(expr_list) == 1
    
    def _handle_first_order_pde(self, expr_list):
        """处理一阶PDE"""
        time_var = self.analyzer_result.get_time_var()
        time_derivatives = self.analyzer_result.get_time_derivatives()
        
        # 获取在表达式中实际出现的函数
        expr_de_func_list = []
        for expr in expr_list:
            meta_exprs = split_expression_meta(expr)
            for time_de_item in time_derivatives.keys():
                if time_de_item in meta_exprs:
                    func = time_de_item.args[0]
                    if func not in expr_de_func_list:
                        expr_de_func_list.append(func)
        
        if expr_de_func_list:
            separate_diff = expr_de_func_list[0].diff(time_var, 1)
            try:
                separate_expr = solve(sympy.Eq(expr_list[0], 0), separate_diff)[0]
                eq_separate = sympy.Eq(separate_diff, separate_expr)
                self.standardized_expressions.append(eq_separate)
            except (IndexError, Exception) as e:
                raise ValueError(f"无法分离导数项 {separate_diff}: {e}")
    
    def _handle_higher_order_pde(self, expr_list):
        """处理高阶PDE"""
        time_var = self.analyzer_result.get_time_var()
        time_derivatives = self.analyzer_result.get_time_derivatives()
        
        # 构建每个表达式中函数的最高阶数字典
        expr_func_de_order_dict = {}
        for expr in expr_list:
            meta_exprs = split_expression_meta(expr)
            
            # 遍历时间导数项，找到在当前表达式中的函数及其阶数
            for time_de_item, order_value in self.time_de_order_dict.items():
                if time_de_item in meta_exprs:
                    func = time_de_item.args[0]
                    current_max_order = expr_func_de_order_dict.get(func, 0)
                    expr_func_de_order_dict[func] = max(current_max_order, order_value)
        
        expr_de_func_list = list(expr_func_de_order_dict.keys())
        
        # 创建替代字典和替代方程
        cnt_idx = 0
        for obj_core_func in expr_de_func_list:
            for index in range(0, expr_func_de_order_dict[obj_core_func] - 1):
                if index + cnt_idx < len(self.sub_obj):
                    # 构建相应的函数替代表达式
                    core_func_args = obj_core_func.args
                    sub_obj_func = self.sub_obj[index + cnt_idx](*core_func_args)
                    self.sub_obj[index + cnt_idx] = sub_obj_func  # 更新sub_obj
                    
                    eq_subs = sympy.Eq(
                        obj_core_func.diff(time_var, index + 1), 
                        sub_obj_func
                    ).subs(self.subs_dict)
                    
                    self.subs_dict[obj_core_func.diff(time_var, index + 1)] = sub_obj_func
                    self.standardized_expressions.append(eq_subs)
                    
            cnt_idx += (expr_func_de_order_dict[obj_core_func] - 1)
        
        # 对于每个表达式，使用替代字典替代所有的导数项
        for expr in expr_list:
            for obj_core_func in expr_de_func_list:
                separate_diff = obj_core_func.diff(time_var, expr_func_de_order_dict[obj_core_func])
                
                try:
                    separate_expr = solve(sympy.Eq(expr, 0), separate_diff)
                    
                    if len(separate_expr) == 0:  # 如果无法分离，跳过
                        continue
                    else:  # 如果分离成功，使用subs_dict进行替换
                        eq_separate = sympy.Eq(separate_diff, separate_expr[0]).subs(self.subs_dict)
                        self.standardized_expressions.append(eq_separate)
                        
                except (IndexError, Exception):
                    continue

    def build_solvable_format(self) -> Dict:
        """
        构建面向py-pde的可求解格式
        
        Returns:
            可求解格式字典
        """
        if not self.standardized_expressions:
            return {}
        
        # 右侧替代字典
        rhs_subs_dict = {}

        # 1) 替代函数项：将 Y_i(...) 形式替换为其函数名，便于构建字符串 RHS
        for sub_item in self.sub_obj:
            if hasattr(sub_item, 'func') and hasattr(sub_item, 'args'):
                func_str, _ = split_func_vars(sub_item)
                rhs_subs_dict[sub_item] = str(func_str)

        # 2) 空间导数项映射：优先使用 py-pde 原生算子 d_dx/d_dy/d2_dx2/d2_dy2；
        #    对混合偏导（如 u_xy）构造嵌套形式 d_dy(d_dx(u))
        spatial_var_list = self.analyzer_result.get_spatial_var_list()

        def _axis_from_symbol(sym_str: str) -> str:
            """根据表达式中的空间变量名映射到算子后缀（'x' 或 'y'）。
            若表达式变量并非标准名（x/y），按 spatial_var_order 顺序映射到 x/y。"""
            try:
                idx = spatial_var_list.index(sym_str)
                return 'x' if idx == 0 else 'y'
            except ValueError:
                # 直接使用原名（若为标准轴），否则退化到 x
                return sym_str if sym_str in ('x', 'y') else 'x'

        from sympy import Derivative as _SymDerivative

        for space_de_item, order_value in self.space_de_order_dict.items():
            # 提取底层函数名，例如 u
            _, func_str, _ = split_derivative_components(space_de_item)
            base_name = func_str[0]

            # SymPy 将 d^2/dx^2 表示为 variables=(x, x)，混合偏导如 d^2/(dx dy) 为 (x, y)
            vars_seq = list(space_de_item.variables)

            if len(vars_seq) == 1:
                # 一阶偏导数 d_dx(u)
                axis = _axis_from_symbol(str(vars_seq[0]))
                rhs_subs_dict[space_de_item] = f"d_d{axis}({base_name})"
            else:
                # 二阶或更高，分两类：
                # - 同轴重复（x,x） -> d2_dx2(u)
                # - 混合（x,y） -> d_dy(d_dx(u))（按 vars_seq 顺序嵌套）
                if len(set(vars_seq)) == 1:
                    axis = _axis_from_symbol(str(vars_seq[0]))
                    if len(vars_seq) == 2:
                        rhs_subs_dict[space_de_item] = f"d2_d{axis}2({base_name})"
                    else:
                        # 暂不支持高于二阶
                        raise ValueError(f"暂不支持大于2阶的空间偏导数，当前阶数: {len(vars_seq)}")
                else:
                    # 混合偏导：依次应用一阶算子
                    nested = base_name
                    for v in vars_seq:
                        axis = _axis_from_symbol(str(v))
                        nested = f"d_d{axis}({nested})"
                    rhs_subs_dict[space_de_item] = nested
        
        # 遍历标准化PDE列表，分别将等式两边转换为可求解的形式
        # 3) 将 RHS 中的未知函数调用（如 u(x,t)、v(x,y,t)）统一替换为字段名符号（u、v）
        #    避免字符串中出现 'u(x, t)' 导致 py-pde 解析失败
        from sympy.core.function import AppliedUndef
        core_func_names = set()
        for f in self.analyzer_result.core_functions:
            try:
                core_func_names.add(str(getattr(f, 'func', f)))
            except Exception:
                pass

        solvable_dict = {}
        for pde_item in self.standardized_expressions:
            try:
                lhs_sub_str = split_derivative_components(pde_item.lhs)[1][0]
                # 先做导数替换/符号函数替代
                rhs_expr = pde_item.rhs.subs(rhs_subs_dict)
                # 再将未知函数调用替换为对应的符号
                applied_calls = list(rhs_expr.atoms(AppliedUndef)) if hasattr(rhs_expr, 'atoms') else []
                func_call_subs = {}
                for call in applied_calls:
                    name = str(getattr(call, 'func', call))
                    if name in core_func_names:
                        # 用同名符号替换函数调用
                        func_call_subs[call] = sympy.Symbol(name)
                if func_call_subs:
                    rhs_expr = rhs_expr.subs(func_call_subs)
                solvable_dict[str(lhs_sub_str)] = str(rhs_expr)
            except (IndexError, Exception):
                # 无法处理的方程跳过
                continue
        
        self.solvable_format = solvable_dict
        return solvable_dict

    def build_substitution_system(self) -> List[sympy.Eq]:
        """
        构建替代方程组
        
        Returns:
            替代方程组列表
        """
        substitution_eqs = []
        
        for original, substitute in self.subs_dict.items():
            eq = sympy.Eq(original, substitute)
            substitution_eqs.append(eq)
        
        return substitution_eqs

    def separate_highest_order_terms(self) -> List[sympy.Eq]:
        """
        分离最高阶导数项
        
        Returns:
            分离后的方程列表
        """
        highest_order_eqs = []
        max_order = max(self.time_de_order_dict.values()) if self.time_de_order_dict else 0
        
        for eq in self.standardized_expressions:
            # 检查是否包含最高阶时间导数
            if isinstance(eq.lhs, sympy.Derivative):
                if eq.lhs in self.time_de_order_dict:
                    derivative_order = self.time_de_order_dict[eq.lhs]
                    if derivative_order == max_order:
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
        return self.subs_dict.copy()
    
    def get_time_derivative_orders(self) -> Dict:
        """获取时间导数阶数字典"""
        return self.time_de_order_dict.copy()
    
    def get_space_derivative_orders(self) -> Dict:
        """获取空间导数阶数字典"""
        return self.space_de_order_dict.copy()
    
    def get_sub_objects(self) -> List:
        """获取替代对象列表"""
        return self.sub_obj.copy()
    
    def get_solvable_format(self) -> Dict:
        """获取可求解格式"""
        if not self.solvable_format:
            self.build_solvable_format()
        return self.solvable_format.copy()
