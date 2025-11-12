"""
Model adapter interfaces.

Defines the standard interface to convert problem definitions into framework-specific models.
"""

from abc import ABC, abstractmethod
from typing import Any
from .problem_definition import IProblemDefinition


class IModelAdapter(ABC):
    """Model adapter interface."""

    @abstractmethod
    def convert(self) -> Any:
        """
        Convert the problem definition to a model for a specific framework.

        Returns:
            The converted model object.
        """
        pass

    @abstractmethod
    def get_target_framework(self) -> str:
        """
        Return the target framework name (e.g., 'pyomo', 'pymoo').
        """
        pass

    @abstractmethod
    def validate_problem(self, problem: IProblemDefinition) -> bool:
        """
        Validate whether the problem is suitable for this adapter.

        Args:
            problem: Problem definition instance.

        Returns:
            True if convertible, else False.
        """
        pass
