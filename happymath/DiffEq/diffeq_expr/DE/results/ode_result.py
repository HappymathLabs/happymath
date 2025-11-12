"""
ODE结果类
封装ODE处理结果，提供与现有ODEModule兼容的接口
"""

from typing import List, Dict
import sympy

from ...base.abstract_result import AbstractResult


class ODEResult(AbstractResult):
    """ODE结果类"""
    
    def __init__(self, analyzer_result, symbol_manager_result, standardizer_result):
        """
        初始化ODE结果类
        
        Args:
            analyzer_result: 分析器结果
            symbol_manager_result: 符号管理器结果
            standardizer_result: 标准化器结果
        """
        super().__init__()
        
        # 存储组件结果
        self._analyzer_result = analyzer_result
        self._symbol_manager_result = symbol_manager_result
        self._standardizer_result = standardizer_result
        
        # 从standardizer_result注入核心数据
        self.standardized_expressions = standardizer_result.standardized_expressions
        self.substitution_dict = standardizer_result.substitution_dict
        
        # 为向后兼容性设置的属性
        self.Y_symbols: List[sympy.Symbol] = symbol_manager_result.get_generated_symbols()
        self.subs_vars_dict: Dict = standardizer_result.substitution_dict.copy()
        self.undetermined_terms: List = self._compute_undetermined_terms()
        self.system_info: Dict = self._compute_system_info()
        
    def _compute_undetermined_terms(self) -> List:
        """计算未定项"""
        undetermined_terms = []
        
        # 添加替代字典中的导数项
        for und_der in self.substitution_dict.keys():
            undetermined_terms.append(und_der)
            
        # 添加核心函数
        if hasattr(self._analyzer_result, 'core_functions'):
            for und_func in self._analyzer_result.core_functions:
                undetermined_terms.append(und_func)
                
        return undetermined_terms
    
    def _compute_system_info(self) -> Dict:
        """计算系统信息"""
        info = {
            'is_system': (hasattr(self._analyzer_result, 'is_system_ode') and 
                         self._analyzer_result.is_system_ode()),
            'expression_type': self._analyzer_result.expression_type,
            'is_linear': self._analyzer_result.is_linear,
            'order': self._analyzer_result.expression_order,
            'num_equations': len(self.standardized_expressions),
            'num_functions': len(self._analyzer_result.core_functions) if hasattr(self._analyzer_result, 'core_functions') else 0,
            'num_variables': len(self._analyzer_result.core_symbols) if hasattr(self._analyzer_result, 'core_symbols') else 0
        }
        
        return info

    def to_dict(self) -> Dict:
        """
        序列化为字典
        
        Returns:
            序列化后的字典
        """
        return {
            'standardized_expressions': self.standardized_expressions,
            'Y_symbols': self.Y_symbols,
            'substitution_dict': self.substitution_dict,
            'subs_vars_dict': self.subs_vars_dict,
            'undetermined_terms': self.undetermined_terms,
            'system_info': self.system_info,
            'expression_type': 'ODE',
            'is_valid': self.is_valid_result()
        }

    def get_undetermined_terms(self) -> List:
        """
        获取未定项列表
        
        Returns:
            未定项列表
        """
        return self.undetermined_terms.copy()

    def is_valid_result(self) -> bool:
        """
        验证结果有效性
        
        Returns:
            是否为有效结果
        """
        # 基本检查
        if not isinstance(self.standardized_expressions, list):
            return False
            
        if not self.standardized_expressions:
            return False
            
        # 检查每个方程是否为Equality类型
        for expr in self.standardized_expressions:
            if not isinstance(expr, sympy.Eq):
                return False
                
        return True

    def get_solver_interface_data(self) -> Dict:
        """
        获取面向求解器的接口数据
        与现有ODEModule接口兼容
        
        Returns:
            求解器接口数据字典
        """
        return {
            'stand_ode_list': self.standardized_expressions,
            'Y_symbols': self.Y_symbols,
            'subs_vars_dict': self.subs_vars_dict,
            'undeter_terms': self.undetermined_terms,
            'system_info': self.system_info
        }
    
    # 为向后兼容性添加的属性访问方法
    def get_standard_equations(self) -> List[sympy.Eq]:
        """获取标准化方程列表（向后兼容）"""
        return self.standardized_expressions
    
    def get_Y_symbols(self) -> List[sympy.Symbol]:
        """获取Y符号列表（向后兼容）"""
        return self.Y_symbols
    
    def get_substitution_dict(self) -> Dict:
        """获取替换字典（向后兼容）"""
        return self.substitution_dict.copy()
    
    def get_subs_vars_dict(self) -> Dict:
        """获取替代变量字典（向后兼容）"""
        return self.subs_vars_dict.copy()
    
    def get_system_info(self) -> Dict:
        """获取系统信息"""
        return self.system_info.copy()
    
    def is_system_ode(self) -> bool:
        """判断是否为ODE方程组"""
        return self.system_info.get('is_system', False)
    
    def is_linear_ode(self) -> bool:
        """判断是否为线性ODE"""
        return self.system_info.get('is_linear', False)
    
    def get_order(self) -> int:
        """获取ODE阶数"""
        return self.system_info.get('order', 0)
    
    def get_num_equations(self) -> int:
        """获取方程数量"""
        return self.system_info.get('num_equations', 0)
    
    def summary(self) -> str:
        """
        返回结果摘要信息
        
        Returns:
            摘要字符串
        """
        info = self.system_info
        
        summary_parts = [
            f"ODE结果摘要:",
            f"  - 类型: {'方程组' if info.get('is_system', False) else '单方程'}",
            f"  - 线性性: {'线性' if info.get('is_linear', False) else '非线性'}",
            f"  - 阶数: {info.get('order', 0)}",
            f"  - 标准化方程数: {info.get('num_equations', 0)}",
            f"  - 核心函数数: {info.get('num_functions', 0)}",
            f"  - 替代符号数: {len(self.Y_symbols)}",
            f"  - 未定项数: {len(self.undetermined_terms)}"
        ]
        
        return "\n".join(summary_parts)