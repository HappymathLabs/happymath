"""
求解器接口

定义了优化求解器的标准接口，支持不同类型的求解器实现。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
from .problem_definition import IProblemDefinition


class ISolver(ABC):
    """求解器接口"""

    @abstractmethod
    def solve(
        self,
        solver: Optional[Union[str, List[str]]] = None,
        use_auto_solvers: bool = True,
        max_solvers: Union[int, str] = 3
    ) -> List[Dict[str, Any]]:
        """
        求解优化问题

        Args:
            solver: 求解器设置
                - None: 自动根据问题类型选择求解器
                - str: 使用指定的单个求解器
                - list: 使用指定的多个求解器
            use_auto_solvers: 是否使用多个求解器
            max_solvers: 最大求解器数量

        Returns:
            求解结果字典列表
        """
        pass

    @abstractmethod
    def get_available_solvers(self) -> List[str]:
        """获取可用求解器列表"""
        pass

    @abstractmethod
    def get_solver_type(self) -> str:
        """获取求解器类型 ('pyomo' 或 'pymoo')"""
        pass


class ISolverFactory(ABC):
    """求解器工厂接口"""

    @abstractmethod
    def create_solvers_for(self, problem: IProblemDefinition) -> List[ISolver]:
        """
        为给定问题创建合适的求解器列表

        Args:
            problem: 问题定义

        Returns:
            求解器列表
        """
        pass

    @abstractmethod
    def get_supported_problem_types(self) -> List[str]:
        """获取支持的问题类型列表"""
        pass