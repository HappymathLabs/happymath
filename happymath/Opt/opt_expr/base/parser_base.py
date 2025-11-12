"""
Abstract parser base class.

Defines a standard interface to convert SymPy expressions into executable forms.
"""

from abc import ABC, abstractmethod
from typing import Any, List


class ParserBase(ABC):
    """Abstract parser base class."""

    @abstractmethod
    def parse(self) -> Any:
        """
        Parse a SymPy expression into an executable object.

        Returns:
            Any: Executable object (e.g., a lambda function).
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate expression correctness.

        Returns:
            bool: Whether the expression is valid.
        """
        pass
