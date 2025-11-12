"""
Abstract analyzer base class.

Defines the standard interfaces for all expression analyzers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class AnalyzerBase(ABC):
    """Abstract expression analyzer base class."""

    def __init__(self, sympy_obj, value_range='real', **kwargs):
        """
        Initialize the analyzer.

        Args:
            sympy_obj: SymPy expression or list of expressions.
            value_range: Symbol assumptions ('real', 'complex', 'integer', etc.).
            **kwargs: Extra parameters.
        """
        self.sympy_obj = sympy_obj
        self.value_range = value_range
        self._analysis_cache = {}  # cache analysis results
        self._analyzed = False

    @abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """
        Run analysis and return a result dictionary.

        Returns:
            Dict[str, Any]: Analysis results.
        """
        pass

    @abstractmethod
    def get_symbols(self):
        """
        Get all symbolic variables.

        Returns:
            set: Set of symbols.
        """
        pass

    def _cache_result(self, key: str, value: Any):
        """Cache analysis results."""
        self._analysis_cache[key] = value

    def _get_cached_result(self, key: str, default=None):
        """Get cached analysis result."""
        return self._analysis_cache.get(key, default)

    def _clear_cache(self):
        """Clear cache and reset analysis flag."""
        self._analysis_cache.clear()
        self._analyzed = False
