"""
模型适配器接口

定义了将问题定义转换为特定框架模型的标准接口。
"""

from abc import ABC, abstractmethod
from typing import Any
from .problem_definition import IProblemDefinition


class IModelAdapter(ABC):
    """模型适配器接口"""

    @abstractmethod
    def convert(self) -> Any:
        """
        将问题定义转换为特定框架的模型

        Returns:
            转换后的模型对象
        """
        pass

    @abstractmethod
    def get_target_framework(self) -> str:
        """
        获取目标框架名称

        Returns:
            框架名称 ('pyomo', 'pymoo', 等)
        """
        pass

    @abstractmethod
    def validate_problem(self, problem: IProblemDefinition) -> bool:
        """
        验证问题是否适合当前适配器

        Args:
            problem: 问题定义

        Returns:
            是否适合转换
        """
        pass