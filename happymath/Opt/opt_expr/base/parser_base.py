"""
解析器抽象基类

定义将SymPy表达式转换为可执行形式的标准接口。
"""

from abc import ABC, abstractmethod
from typing import Any, List


class ParserBase(ABC):
    """解析器抽象基类"""

    @abstractmethod
    def parse(self) -> Any:
        """
        将SymPy表达式解析为可执行形式

        Returns:
            Any: 解析后的可执行对象(如lambda函数)
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """
        验证表达式是否有效

        Returns:
            bool: 表达式是否有效
        """
        pass
