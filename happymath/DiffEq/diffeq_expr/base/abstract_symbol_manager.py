"""
Abstract base class for symbol managers.
Responsibilities:
  - Compute substitution requirements and generate substitute symbols/objects
  - Maintain mappings and validate conflicts
  - Provide utilities to split expression components
"""

from abc import ABC, abstractmethod
from typing import List, Dict


class AbstractSymbolManager(ABC):
    """Abstract symbol manager."""
    
    def __init__(self, analyzer_result):
        """
        Initialize the symbol manager.

        Args:
            analyzer_result: Analyzer result object.
        """
        self.analyzer_result = analyzer_result
        self.symbol_mappings: Dict = {}
        self.substitute_symbols: List = []

    @abstractmethod
    def generate_substitute_symbols(self, count: int, prefix: str = 'Y', mode: str = 'symbol') -> List:
        """
        Generate substitute symbols/objects.

        Args:
            count: Number of symbols to generate.
            prefix: Symbol prefix.
            mode: 'symbol' or 'function'.

        Returns:
            List of substitute symbols/objects.
        """
        pass

    @abstractmethod
    def create_symbol_mappings(self) -> Dict:
        """
        Create symbol mapping dictionary.

        Returns:
            Mapping of symbols.
        """
        pass

    @abstractmethod
    def validate_symbol_conflicts(self, symbols: List) -> bool:
        """
        Validate symbol conflicts.

        Args:
            symbols: Symbols to check.

        Returns:
            True if conflicts exist, else False.
        """
        pass

    @abstractmethod
    def get_substitution_count(self) -> int:
        """
        Return number of required substitute symbols.
        """
        pass

    @abstractmethod
    def split_expression_components(self, expr) -> Dict:
        """
        Split expression into components.

        Args:
            expr: Expression to split.

        Returns:
            Dictionary of components.
        """
        pass
