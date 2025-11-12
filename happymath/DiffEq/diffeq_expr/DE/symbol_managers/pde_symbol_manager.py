"""
PDE符号管理器
从PDEModule中迁移的符号生成和管理功能
"""

from typing import List, Dict, Union
import sympy
from sympy.utilities.iterables import iterable

from ...base.abstract_symbol_manager import AbstractSymbolManager
from ...utils import split_expression_meta, split_func_vars


class PDESymbolManager(AbstractSymbolManager):
    """PDE符号管理器"""
    
    def __init__(self, analyzer_result):
        """
        初始化PDE符号管理器
        
        Args:
            analyzer_result: PDE分析器结果对象
        """
        super().__init__(analyzer_result)
        self._generated_symbols = []
        self._sub_objects = []  # 存储函数型替代对象
        
    def generate_substitute_symbols(self, count: int, prefix: str = 'Y', mode: str = 'function') -> List:
        """
        生成替代符号/对象
        对于PDE，主要生成函数型替代对象 Y_i(x, y, t)
        
        Args:
            count: 需要生成的符号数量
            prefix: 符号前缀
            mode: 生成模式，'symbol' 或 'function'
            
        Returns:
            替代符号/对象列表
        """
        if count <= 0:
            return []
            
        # 获取表达式中已存在的符号
        existing_symbols = self._get_existing_symbols()
        
        symbols = []
        start_index = 1
        
        # 生成指定数量的符号
        while len(symbols) < count:
            if mode == 'function':
                # 生成函数型替代对象
                candidate_symbols = [sympy.Function(f"{prefix}_{start_index + i}") for i in range(count - len(symbols))]
            elif mode == 'symbol':
                # 生成符号型
                candidate_symbols = [sympy.Symbol(f"{prefix}_{start_index + i}") for i in range(count - len(symbols))]
            else:
                raise ValueError("Invalid mode. Please choose from 'symbol', 'function'.")
            
            # 检查符号冲突
            non_conflicting = []
            for sym in candidate_symbols:
                if not self._has_symbol_conflict(sym, existing_symbols):
                    non_conflicting.append(sym)
                    existing_symbols.add(sym)  # 更新已存在符号集合
                    
                if len(symbols) + len(non_conflicting) >= count:
                    break
                    
            symbols.extend(non_conflicting)
            start_index += len(candidate_symbols)
            
            # 防止无限循环
            if start_index > 1000:
                raise RuntimeError(f"在1000次尝试后无法生成{count}个唯一符号")
        
        # 只取需要的数量
        result = symbols[:count]
        self._generated_symbols = result
        return result
    
    def _get_existing_symbols(self) -> set:
        """获取表达式中已存在的所有符号"""
        existing = set()
        
        if iterable(self.analyzer_result.sympy_obj):
            for expr in self.analyzer_result.sympy_obj:
                existing.update(expr.atoms(sympy.Symbol))
                existing.update(expr.atoms(sympy.Function))
        else:
            existing.update(self.analyzer_result.sympy_obj.atoms(sympy.Symbol))
            existing.update(self.analyzer_result.sympy_obj.atoms(sympy.Function))
        
        return existing
    
    def _has_symbol_conflict(self, symbol, existing_symbols: set) -> bool:
        """检查符号是否与现有符号冲突"""
        # 检查符号名称冲突
        symbol_name = str(symbol)
        for existing in existing_symbols:
            if str(existing) == symbol_name:
                return True
        return False

    def get_substitution_count(self) -> int:
        """
        获取需要的替代符号数量
        规则：每个函数关于时间导数最高阶数 - 1 的和
        
        Returns:
            替代符号数量
        """
        time_derivatives = self.analyzer_result.get_time_derivatives()
        
        # 计算每个函数关于时间的最高阶数
        func_max_order = {}
        for de_item, order_value in time_derivatives.items():
            func = de_item.args[0]  # 获取被微分的函数
            func_max_order[func] = max(func_max_order.get(func, 0), order_value)
        
        # 计算替代符号数量为每个函数最高阶数-1的和
        sub_obj_num = 0
        for func, max_order in func_max_order.items():
            sub_obj_num += max(0, max_order - 1)  # 确保不会出现负数
        
        return sub_obj_num

    def create_symbol_mappings(self) -> Dict:
        """
        创建符号映射关系
        
        Returns:
            符号映射字典
        """
        if not self._generated_symbols:
            count = self.get_substitution_count()
            self.generate_substitute_symbols(count, mode='function')
        
        # 为PDE创建映射关系
        mappings = self._create_pde_mappings()
        
        self.symbol_mappings = mappings
        return mappings
    
    def _create_pde_mappings(self) -> Dict:
        """为PDE创建映射"""
        mappings = {}
        
        time_derivatives = self.analyzer_result.get_time_derivatives()
        time_var = self.analyzer_result.get_time_var()
        
        if not time_var:
            return mappings
        
        # 计算每个函数关于时间的最高阶数
        expr_func_de_order_dict = {}
        
        # 遍历所有时间导数项，找到每个函数的最高阶数
        for time_de_item, order_value in time_derivatives.items():
            func = time_de_item.args[0]  # 正在被微分的函数
            current_max_order = expr_func_de_order_dict.get(func, 0)
            expr_func_de_order_dict[func] = max(current_max_order, order_value)
        
        # 为每个函数的时间导数创建替代映射
        cnt_idx = 0
        for obj_core_func, max_order in expr_func_de_order_dict.items():
            for index in range(0, max_order - 1):
                if index + cnt_idx < len(self._generated_symbols):
                    # 构建具有相同参数的函数替代表达式
                    core_func_args = obj_core_func.args
                    sub_obj_func = self._generated_symbols[index + cnt_idx](*core_func_args)
                    
                    # 创建映射
                    derivative = obj_core_func.diff(time_var, index + 1)
                    mappings[derivative] = sub_obj_func
                    
            cnt_idx += (max_order - 1)
        
        return mappings

    def validate_symbol_conflicts(self, symbols: List) -> bool:
        """
        验证符号冲突
        
        Args:
            symbols: 待验证的符号列表
            
        Returns:
            是否存在冲突（True表示无冲突）
        """
        existing = self._get_existing_symbols()
        
        for symbol in symbols:
            if self._has_symbol_conflict(symbol, existing):
                return False
            existing.add(symbol)  # 添加到已存在符号集合中，检查内部冲突
                
        return True

    def split_expression_components(self, expr) -> Dict:
        """
        分解表达式组件
        
        Args:
            expr: 表达式
            
        Returns:
            分解后的组件字典
        """
        components = {
            'functions': list(expr.atoms(sympy.Function)),
            'symbols': list(expr.free_symbols),
            'derivatives': list(expr.atoms(sympy.Derivative)),
            'time_derivatives': [],
            'spatial_derivatives': [],
            'constants': [],
            'meta_expressions': split_expression_meta(expr)
        }
        
        # 分离时间和空间导数
        for deriv in components['derivatives']:
            if str(deriv.args[1][0]) == "t":
                components['time_derivatives'].append(deriv)
            else:
                components['spatial_derivatives'].append(deriv)
        
        # 分离常数
        for symbol in expr.free_symbols:
            if symbol.is_constant():
                components['constants'].append(symbol)
        
        return components
    
    def get_generated_symbols(self) -> List:
        """获取已生成的符号列表"""
        return self._generated_symbols.copy()
    
    def get_sub_objects(self) -> List:
        """获取替代对象列表"""
        return self._sub_objects.copy()
    
    def set_sub_objects(self, sub_objects: List):
        """设置替代对象列表"""
        self._sub_objects = sub_objects.copy()
    
    def clear_generated_symbols(self):
        """清除已生成的符号"""
        self._generated_symbols.clear()
        self._sub_objects.clear()
        self.symbol_mappings.clear()
    
    def create_function_substitutes(self, core_func_args_list: List[tuple]) -> List:
        """
        创建具有特定参数的函数替代对象
        
        Args:
            core_func_args_list: 核心函数参数列表
            
        Returns:
            函数替代对象列表
        """
        if not self._generated_symbols:
            return []
        
        function_subs = []
        for i, func_symbol in enumerate(self._generated_symbols):
            if i < len(core_func_args_list):
                args = core_func_args_list[i]
                function_subs.append(func_symbol(*args))
            else:
                # 如果没有足够的参数信息，使用默认参数
                default_args = self.analyzer_result.core_symbols
                function_subs.append(func_symbol(*default_args))
        
        self._sub_objects = function_subs
        return function_subs