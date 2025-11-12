"""
问题定义接口

定义了优化问题的标准接口，任何实现了这个接口的类都可以被
适配器和求解器使用，从而实现松耦合的设计。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from sympy import Symbol


class IProblemDefinition(ABC):
    """优化问题定义接口"""

    @property
    @abstractmethod
    def objective_funcs(self) -> List[Any]:
        """目标函数列表"""
        pass

    @property
    @abstractmethod
    def objective_exprs(self) -> List[Any]:
        """目标表达式列表"""
        pass

    @property
    @abstractmethod
    def senses(self) -> List[str]:
        """优化方向列表 ('min' 或 'max')"""
        pass

    @property
    @abstractmethod
    def parsed_constraints(self) -> List[Any]:
        """解析后的约束条件列表"""
        pass

    @property
    @abstractmethod
    def all_symbols(self) -> set:
        """所有符号的集合"""
        pass

    @property
    @abstractmethod
    def sorted_symbols(self) -> List[Symbol]:
        """排序后的符号列表"""
        pass

    @property
    @abstractmethod
    def variable_bounds(self) -> Tuple[List[float], List[float]]:
        """变量边界 (下界列表, 上界列表)"""
        pass

    @abstractmethod
    def has_integer_variables(self) -> bool:
        """是否包含整数变量"""
        pass

    @abstractmethod
    def get_pyomo_problem_type(self) -> str:
        """获取Pyomo问题类型"""
        pass

    @abstractmethod
    def get_pymoo_problem_type(self) -> Dict[str, Any]:
        """获取Pymoo问题类型字典"""
        pass
