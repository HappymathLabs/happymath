"""
Variable manager.

Manages all decision variable symbols:
- Collect symbols
- Provide stable ordering
- Build symbol-to-index mapping
"""

from typing import Set, List, Dict
from sympy import Symbol


class VariableManager:
    """Manager for decision variable symbols."""

    def __init__(self, obj_analyzer, con_analyzer=None, extra_symbols=None, exclude_symbols=None):
        """
        Initialize variable manager.

        Args:
            obj_analyzer: ObjectiveAnalyzer instance.
            con_analyzer: Optional ConstraintAnalyzer instance.
        """
        self.obj_analyzer = obj_analyzer
        self.con_analyzer = con_analyzer
        self._extra_symbols = list(extra_symbols or [])
        self._exclude_symbols = set(s for s in (exclude_symbols or []) if s is not None)

        self._all_symbols = None
        self._sorted_symbols = None
        self._symbol_to_index = None

    def collect_all_symbols(self) -> Set[Symbol]:
        """
        Collect all decision variable symbols.

        Returns:
            Set[Symbol]: Set of all symbols.
        """
        if self._all_symbols is not None:
            return self._all_symbols

        # From objectives
        all_symbols = self.obj_analyzer.get_symbols().copy()

        # From constraints
        if self.con_analyzer:
            all_symbols.update(self.con_analyzer.get_symbols())

        # Merge external symbols (e.g., control coefficients, initial symbols)
        for s in self._extra_symbols:
            try:
                if s is not None:
                    all_symbols.add(s)
            except Exception:
                continue

        # Exclude independent domain variables (e.g., t)
        if self._exclude_symbols:
            all_symbols = {s for s in all_symbols if s not in self._exclude_symbols}

        self._all_symbols = all_symbols
        return all_symbols

    def get_sorted_symbols(self) -> List[Symbol]:
        """
        Return symbols sorted by string representation.

        Returns:
            List[Symbol]
        """
        if self._sorted_symbols is not None:
            return self._sorted_symbols

        all_symbols = self.collect_all_symbols()
        self._sorted_symbols = sorted(list(all_symbols), key=lambda s: str(s))
        return self._sorted_symbols

    def get_symbol_to_index_mapping(self) -> Dict[Symbol, int]:
        """
        Return mapping from symbol to index.

        Returns:
            Dict[Symbol, int]
        """
        if self._symbol_to_index is not None:
            return self._symbol_to_index

        sorted_symbols = self.get_sorted_symbols()
        self._symbol_to_index = {sym: i for i, sym in enumerate(sorted_symbols)}
        return self._symbol_to_index

    @property
    def all_symbols(self) -> Set[Symbol]:
        """Return all symbols."""
        return self.collect_all_symbols()

    @property
    def sorted_symbols(self) -> List[Symbol]:
        """Return sorted symbols."""
        return self.get_sorted_symbols()

    @property
    def symbol_to_index(self) -> Dict[Symbol, int]:
        """Return symbol-to-index mapping."""
        return self.get_symbol_to_index_mapping()

    @property
    def n_variables(self) -> int:
        """Return number of variables."""
        return len(self.sorted_symbols)
