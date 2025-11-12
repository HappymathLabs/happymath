"""
标准化器抽象基类
职责：
  - 执行标准化
  - 构建替代方程组/字典
  - 分离最高阶导数
"""

from abc import ABC, abstractmethod
from typing import List, Dict
import sympy


class AbstractStandardizer(ABC):
    """标准化器抽象基类"""
    
    def __init__(self, analyzer_result, symbol_manager_result):
        """
        初始化标准化器
        
        Args:
            analyzer_result: 分析器结果对象
            symbol_manager_result: 符号管理器结果对象
        """
        self.analyzer_result = analyzer_result
        self.symbol_manager_result = symbol_manager_result
        self.standardized_expressions: List[sympy.Eq] = []
        self.substitution_dict: Dict = {}

    @abstractmethod
    def standardize(self):
        """
        执行标准化处理
        
        Returns:
            标准化器对象本身（链式调用）
        """
        pass

    @abstractmethod
    def build_substitution_system(self) -> List[sympy.Eq]:
        """
        构建替代方程组
        
        Returns:
            替代方程组列表
        """
        pass

    @abstractmethod
    def separate_highest_order_terms(self) -> List[sympy.Eq]:
        """
        分离最高阶导数项
        
        Returns:
            分离后的方程列表
        """
        pass

    @abstractmethod
    def create_standard_form(self) -> List[sympy.Eq]:
        """
        创建标准形式
        
        Returns:
            标准形式方程列表
        """
        pass

    @abstractmethod
    def get_substitution_dict(self) -> Dict:
        """
        获取替代字典
        
        Returns:
            替代字典
        """
        pass