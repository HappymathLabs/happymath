"""
表达式分析器抽象基类
职责：
  - 验证表达式有效性
  - 提取核心函数、符号、导数等元信息
  - 判断表达式线性等属性
"""

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Set
import sympy


class AbstractExpressionAnalyzer(ABC):
    """表达式分析器抽象基类"""
    
    def __init__(self, sympy_obj: Union[sympy.Expr, List], value_range: str = "real"):
        """
        初始化表达式分析器
        
        Args:
            sympy_obj: 微分方程表达式或表达式列表
            value_range: 变量取值范围
        """
        self.sympy_obj = sympy_obj
        self.value_range = value_range
        self._cache: Dict[str, object] = {}

    @abstractmethod
    def is_valid_expression(self) -> bool:
        """
        验证表达式有效性
        
        Returns:
            是否为有效表达式
        """
        pass

    @property
    @abstractmethod
    def expression_type(self) -> str:
        """
        表达式类型
        
        Returns:
            表达式类型，如 "ODE" | "PDE"
        """
        pass

    @property
    @abstractmethod
    def is_linear(self) -> bool:
        """
        判断表达式是否线性
        
        Returns:
            是否为线性表达式
        """
        pass

    @property
    @abstractmethod
    def core_functions(self) -> List[sympy.Function]:
        """
        提取核心函数列表
        
        Returns:
            核心函数列表
        """
        pass

    @property
    @abstractmethod
    def core_symbols(self) -> List[sympy.Symbol]:
        """
        提取核心符号列表
        
        Returns:
            核心符号列表
        """
        pass

    @property
    @abstractmethod
    def derivative_orders(self) -> Dict[sympy.Derivative, int]:
        """
        提取导数项与阶数的映射
        
        Returns:
            导数项与阶数的映射字典
        """
        pass

    @property
    @abstractmethod
    def free_constants(self) -> Set[sympy.Symbol]:
        """
        提取自由常数集合
        
        Returns:
            自由常数集合
        """
        pass

    @property
    @abstractmethod
    def expression_order(self) -> int:
        """
        获取表达式最大阶数
        
        Returns:
            最大阶数
        """
        pass

    @property
    @abstractmethod
    def core_func_symbol_mapping(self) -> Dict:
        """
        获取核心函数与符号的映射关系
        
        Returns:
            函数与符号的映射字典
        """
        pass
    
    def invalidate_cache(self):
        """使缓存失效"""
        self._cache.clear()
    
    def get_cached_value(self, key: str):
        """获取缓存值"""
        return self._cache.get(key)
    
    def set_cached_value(self, key: str, value):
        """设置缓存值"""
        self._cache[key] = value