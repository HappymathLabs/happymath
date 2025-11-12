"""
ODE符号管理器
从ODEModule中迁移的符号生成和管理功能
"""

from typing import List, Dict, Union
import sympy
from sympy.utilities.iterables import iterable

from ...base.abstract_symbol_manager import AbstractSymbolManager
from ...utils import split_expression_meta


class ODESymbolManager(AbstractSymbolManager):
    """ODE符号管理器"""
    
    def __init__(self, analyzer_result):
        """
        初始化ODE符号管理器
        
        Args:
            analyzer_result: ODE分析器结果对象
        """
        super().__init__(analyzer_result)
        self._generated_symbols = []
        
    def generate_substitute_symbols(self, count: int, prefix: str = 'Y', mode: str = 'symbol') -> List:
        """
        生成替代符号
        迁移ODEModule中的唯一符号生成策略
        
        Args:
            count: 需要生成的符号数量
            prefix: 符号前缀
            mode: 生成模式，'symbol' 或 'function'
            
        Returns:
            替代符号列表
        """
        if count <= 0:
            return []
            
        # 获取表达式中已存在的符号
        existing_symbols = self._get_existing_symbols()
        
        symbols = []
        start_index = 1
        
        # 生成指定数量的符号
        while len(symbols) < count:
            if mode == 'symbol':
                # 生成符号型
                candidate_symbols = [sympy.Symbol(f"{prefix}_{start_index + i}") for i in range(count - len(symbols))]
            elif mode == 'function':
                # 生成函数型（暂时用Symbol替代，具体应用时再转换）
                candidate_symbols = [sympy.Function(f"{prefix}_{start_index + i}") for i in range(count - len(symbols))]
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
        规则：所有导数阶数之和减去核心函数数量
        
        Returns:
            替代符号数量
        """
        if hasattr(self.analyzer_result, 'is_system_ode') and self.analyzer_result.is_system_ode():
            return self._get_system_ode_substitution_count()
        else:
            return self._get_single_ode_substitution_count()
    
    def _get_system_ode_substitution_count(self) -> int:
        """获取ODE方程组的替代符号数量（扫描导数项，不依赖classify_sysode）"""
        try:
            der_orders = getattr(self.analyzer_result, 'derivative_orders', {})
        except Exception:
            der_orders = {}

        # 统计每个未知函数的最高阶
        max_order_by_func = {}
        if isinstance(der_orders, dict) and der_orders:
            for der, order in der_orders.items():
                try:
                    base = der.args[0]
                except Exception:
                    continue
                max_order_by_func[base] = max(max_order_by_func.get(base, 0), int(order))

        # 对于每个函数 f，替代符号数量为 max(order(f)-1, 0)
        num_subs = sum(max(v - 1, 0) for v in max_order_by_func.values())
        return max(0, int(num_subs))
    
    def _get_single_ode_substitution_count(self) -> int:
        """获取单个ODE的替代符号数量"""
        # 为了保持与原始ODEModule的兼容性，总是生成order_num个符号
        # 即使对于一阶ODE也是如此，尽管实际可能不会全部使用
        order_num = self.analyzer_result.expression_order
        return order_num  # 修改：原来是 max(0, order_num - 1)

    def create_symbol_mappings(self) -> Dict:
        """
        创建符号映射关系
        
        Returns:
            符号映射字典
        """
        if not self._generated_symbols:
            count = self.get_substitution_count()
            self.generate_substitute_symbols(count)
        
        # 为ODE创建映射关系
        mappings = {}
        if hasattr(self.analyzer_result, 'is_system_ode') and self.analyzer_result.is_system_ode():
            mappings = self._create_system_mappings()
        else:
            mappings = self._create_single_mappings()
        
        self.symbol_mappings = mappings
        return mappings
    
    def _create_system_mappings(self) -> Dict:
        """为ODE方程组创建映射（扫描导数项，不依赖classify_sysode）"""
        mappings = {}
        core_symbols = self.analyzer_result.core_symbols
        core_funcs = getattr(self.analyzer_result, 'core_functions', [])
        if not core_symbols or not core_funcs:
            return mappings

        core_symbol = core_symbols[0]

        # 统计每个未知函数的最高阶
        try:
            der_orders = getattr(self.analyzer_result, 'derivative_orders', {})
        except Exception:
            der_orders = {}

        max_order_by_func = {}
        if isinstance(der_orders, dict) and der_orders:
            for der, order in der_orders.items():
                try:
                    base = der.args[0]
                except Exception:
                    continue
                if base in core_funcs:
                    max_order_by_func[base] = max(max_order_by_func.get(base, 0), int(order))

        # 依次为每个函数的低阶导数建立映射：f'(t)->Y_1, f''(t)->Y_2, ...（不含最高阶）
        cnt_idx = 0
        for core_func in core_funcs:
            order_v = max_order_by_func.get(core_func, 0)
            for index in range(0, max(order_v - 1, 0)):
                if index + cnt_idx < len(self._generated_symbols):
                    derivative = core_func.diff(core_symbol, index + 1)
                    mappings[derivative] = self._generated_symbols[index + cnt_idx]
            cnt_idx += max(order_v - 1, 0)

        return mappings
    
    def _create_single_mappings(self) -> Dict:
        """为单个ODE创建映射"""
        mappings = {}
        
        if self._generated_symbols and self.analyzer_result.core_functions and self.analyzer_result.core_symbols:
            core_func = self.analyzer_result.core_functions[0]
            core_symbol = self.analyzer_result.core_symbols[0]
            order_num = self.analyzer_result.expression_order
            
            # 为了与原始ODEModule兼容，为每个阶次的导数创建映射（除了最高阶）
            # 但是对于一阶ODE，实际上不创建任何映射，因为原始逻辑也是这样
            if order_num > 1:
                for index in range(order_num - 1):
                    if index < len(self._generated_symbols):
                        derivative = core_func.diff(core_symbol, index + 1)
                        mappings[derivative] = self._generated_symbols[index]
        
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
            'constants': [],
            'meta_expressions': split_expression_meta(expr)
        }
        
        # 分离常数
        for symbol in expr.free_symbols:
            if symbol.is_constant():
                components['constants'].append(symbol)
        
        return components
    
    def get_generated_symbols(self) -> List:
        """获取已生成的符号列表"""
        return self._generated_symbols.copy()
    
    def clear_generated_symbols(self):
        """清除已生成的符号"""
        self._generated_symbols.clear()
        self.symbol_mappings.clear()
