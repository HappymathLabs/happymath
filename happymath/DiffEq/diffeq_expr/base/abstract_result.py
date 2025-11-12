"""
结果类抽象基类
统一封装输出，提供序列化与求解器接口数据
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Set
import sympy


class AbstractResult(ABC):
    """结果类抽象基类"""
    
    def __init__(self):
        """初始化结果类"""
        self.standardized_expressions: List[sympy.Eq] = []
        self.symbol_mappings: Dict = {}
        self.derivative_info: Dict = {}
        self.variable_classification: Dict = {}
        self.constant_info: Set = set()
        self.substitution_dict: Dict = {}

    @abstractmethod
    def to_dict(self) -> Dict:
        """
        序列化为字典
        
        Returns:
            序列化后的字典
        """
        pass

    @abstractmethod
    def get_undetermined_terms(self) -> List:
        """
        获取未定项列表
        
        Returns:
            未定项列表
        """
        pass

    @abstractmethod
    def is_valid_result(self) -> bool:
        """
        验证结果有效性
        
        Returns:
            是否为有效结果
        """
        pass

    @abstractmethod
    def get_solver_interface_data(self) -> Dict:
        """
        获取面向求解器的接口数据
        
        Returns:
            求解器接口数据字典
        """
        pass