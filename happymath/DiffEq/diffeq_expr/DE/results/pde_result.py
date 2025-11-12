"""
PDE结果类
封装PDE处理结果，提供与现有PDEModule兼容的接口
"""

from typing import List, Dict
import sympy

from ...base.abstract_result import AbstractResult


class PDEResult(AbstractResult):
    """PDE结果类"""
    
    def __init__(self, analyzer_result, symbol_manager_result, standardizer_result):
        """
        初始化PDE结果类
        
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
        
        # PDE特有属性
        self.time_derivative_orders: Dict = standardizer_result.get_time_derivative_orders()
        self.space_derivative_orders: Dict = standardizer_result.get_space_derivative_orders()
        self.solvable_format: Dict = standardizer_result.get_solvable_format()
        self.spatial_variables: List[sympy.Symbol] = analyzer_result.spatial_vars
        self.time_variable = analyzer_result.time_var
        self.sub_objects: List = standardizer_result.get_sub_objects()
        
        # 为向后兼容性设置的属性
        self.spatial_var_list = analyzer_result.get_spatial_var_list()
        
    def to_dict(self) -> Dict:
        """
        序列化为字典
        
        Returns:
            序列化后的字典
        """
        return {
            'standardized_expressions': self.standardized_expressions,
            'solvable_format': self.solvable_format,
            'time_derivatives': self.time_derivative_orders,
            'space_derivatives': self.space_derivative_orders,
            'spatial_variables': self.spatial_variables,
            'spatial_var_list': self.spatial_var_list,
            'time_variable': self.time_variable,
            'sub_objects': self.sub_objects,
            'substitution_dict': self.substitution_dict,
            'expression_type': 'PDE',
            'is_valid': self.is_valid_result()
        }

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
        if hasattr(self._analyzer_result, 'core_functions'):
            for und_func in self._analyzer_result.core_functions:
                undetermined_terms.append(und_func)
                
        return undetermined_terms

    def is_valid_result(self) -> bool:
        """
        验证结果有效性
        
        Returns:
            是否为有效结果
        """
        # 基本检查
        if not isinstance(self.standardized_expressions, list):
            return False
        
        # 对于PDE，可以允许空的标准化表达式列表（某些情况下）
        # 但至少需要有时间变量
        if not self.time_variable:
            return False
            
        # 检查每个方程是否为Equality类型（如果有的话）
        for expr in self.standardized_expressions:
            if not isinstance(expr, sympy.Eq):
                return False
                
        return True

    def get_solver_interface_data(self) -> Dict:
        """
        获取面向求解器的接口数据
        与现有PDEModule接口兼容
        
        Returns:
            求解器接口数据字典
        """
        return {
            'stand_pde_list': self.standardized_expressions,
            'to_solvable_pde_dict': self.solvable_format,
            'time_de_order_dict': self.time_derivative_orders,
            'space_de_order_dict': self.space_derivative_orders,
            'sub_obj': self.sub_objects,
            'spatial_variables': self.spatial_variables,
            'time_variable': self.time_variable,
            'spatial_var_list': self.spatial_var_list
        }
    
    # 为向后兼容性添加的属性访问方法
    def get_standard_equations(self) -> List[sympy.Eq]:
        """获取标准化方程列表（向后兼容）"""
        return self.standardized_expressions
    
    def get_solvable_format(self) -> Dict:
        """获取可求解格式（向后兼容）"""
        return self.solvable_format.copy()
    
    def get_time_derivative_orders(self) -> Dict:
        """获取时间导数阶数字典"""
        return self.time_derivative_orders.copy()
    
    def get_space_derivative_orders(self) -> Dict:
        """获取空间导数阶数字典"""
        return self.space_derivative_orders.copy()
    
    def get_sub_objects(self) -> List:
        """获取替代对象列表"""
        return self.sub_objects.copy()
    
    def get_spatial_variables(self) -> List[sympy.Symbol]:
        """获取空间变量列表"""
        return self.spatial_variables.copy()
    
    def get_time_variable(self) -> sympy.Symbol:
        """获取时间变量"""
        return self.time_variable
    
    def get_spatial_var_list(self) -> List[str]:
        """获取空间变量名称列表"""
        return self.spatial_var_list.copy()
    
    def get_substitution_dict(self) -> Dict:
        """获取替换字典"""
        return self.substitution_dict.copy()
    
    def is_linear_pde(self) -> bool:
        """判断是否为线性PDE"""
        return self._analyzer_result.is_linear if hasattr(self._analyzer_result, 'is_linear') else False
    
    def get_order(self) -> int:
        """获取PDE阶数"""
        return self._analyzer_result.expression_order if hasattr(self._analyzer_result, 'expression_order') else 0
    
    def get_num_equations(self) -> int:
        """获取方程数量"""
        return len(self.standardized_expressions)
    
    def get_spatial_dimensions(self) -> int:
        """获取空间维度数"""
        return len(self.spatial_variables)
    
    def has_time_variable(self) -> bool:
        """判断是否包含时间变量"""
        return self.time_variable is not None
    
    def validate_spatial_dimensions(self) -> bool:
        """验证空间维度（当前限制≤2维）"""
        return len(self.spatial_variables) <= 2
    
    def check_mixed_derivatives(self) -> bool:
        """检查是否含混合偏导数"""
        if hasattr(self._analyzer_result, 'check_mixed_derivatives'):
            return self._analyzer_result.check_mixed_derivatives()
        return False
    
    def summary(self) -> str:
        """
        返回结果摘要信息
        
        Returns:
            摘要字符串
        """
        summary_parts = [
            f"PDE结果摘要:",
            f"  - 线性性: {'线性' if self.is_linear_pde() else '非线性'}",
            f"  - 阶数: {self.get_order()}",
            f"  - 标准化方程数: {self.get_num_equations()}",
            f"  - 空间维度: {self.get_spatial_dimensions()}",
            f"  - 空间变量: {', '.join(self.get_spatial_var_list())}",
            f"  - 时间变量: {self.time_variable if self.time_variable else 'None'}",
            f"  - 替代对象数: {len(self.sub_objects)}",
            f"  - 时间导数项数: {len(self.time_derivative_orders)}",
            f"  - 空间导数项数: {len(self.space_derivative_orders)}",
            f"  - 可求解格式项数: {len(self.solvable_format)}",
            f"  - 包含混合偏导数: {'是' if self.check_mixed_derivatives() else '否'}"
        ]
        
        return "\n".join(summary_parts)