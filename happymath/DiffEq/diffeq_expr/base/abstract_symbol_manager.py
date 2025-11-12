"""
符号管理器抽象基类
职责：
  - 计算替代需求与生成替代符号/对象
  - 维护映射与冲突验证
  - 提供表达式组件分解工具
"""

from abc import ABC, abstractmethod
from typing import List, Dict


class AbstractSymbolManager(ABC):
    """符号管理器抽象基类"""
    
    def __init__(self, analyzer_result):
        """
        初始化符号管理器
        
        Args:
            analyzer_result: 分析器结果对象
        """
        self.analyzer_result = analyzer_result
        self.symbol_mappings: Dict = {}
        self.substitute_symbols: List = []

    @abstractmethod
    def generate_substitute_symbols(self, count: int, prefix: str = 'Y', mode: str = 'symbol') -> List:
        """
        生成替代符号/对象
        
        Args:
            count: 需要生成的符号数量
            prefix: 符号前缀
            mode: 生成模式，'symbol' 或 'function'
            
        Returns:
            替代符号/对象列表
        """
        pass

    @abstractmethod
    def create_symbol_mappings(self) -> Dict:
        """
        创建符号映射关系
        
        Returns:
            符号映射字典
        """
        pass

    @abstractmethod
    def validate_symbol_conflicts(self, symbols: List) -> bool:
        """
        验证符号冲突
        
        Args:
            symbols: 待验证的符号列表
            
        Returns:
            是否存在冲突
        """
        pass

    @abstractmethod
    def get_substitution_count(self) -> int:
        """
        获取需要的替代符号数量
        
        Returns:
            替代符号数量
        """
        pass

    @abstractmethod
    def split_expression_components(self, expr) -> Dict:
        """
        分解表达式组件
        
        Args:
            expr: 表达式
            
        Returns:
            分解后的组件字典
        """
        pass