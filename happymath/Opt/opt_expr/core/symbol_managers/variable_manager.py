"""
变量管理器

负责管理优化问题中的所有变量符号，包括：
- 收集所有变量符号
- 管理变量排序（确保一致性）
- 提供符号到索引的映射
"""

from typing import Set, List, Dict
from sympy import Symbol


class VariableManager:
    """变量管理器"""

    def __init__(self, obj_analyzer, con_analyzer=None, extra_symbols=None, exclude_symbols=None):
        """
        初始化变量管理器

        Args:
            obj_analyzer: ObjectiveAnalyzer实例
            con_analyzer: ConstraintAnalyzer实例（可选）
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
        收集所有变量符号

        Returns:
            Set[Symbol]: 所有符号变量的集合
        """
        if self._all_symbols is not None:
            return self._all_symbols

        # 从目标函数收集符号
        all_symbols = self.obj_analyzer.get_symbols().copy()

        # 从约束条件收集符号
        if self.con_analyzer:
            all_symbols.update(self.con_analyzer.get_symbols())

        # 合入外部提供的额外决策变量符号（如控制系数、初值符号等）
        for s in self._extra_symbols:
            try:
                if s is not None:
                    all_symbols.add(s)
            except Exception:
                continue

        # 排除连续域自变量（如 t）等
        if self._exclude_symbols:
            all_symbols = {s for s in all_symbols if s not in self._exclude_symbols}

        self._all_symbols = all_symbols
        return all_symbols

    def get_sorted_symbols(self) -> List[Symbol]:
        """
        获取排序后的符号列表（按字符串表示排序）

        Returns:
            List[Symbol]: 排序后的符号列表
        """
        if self._sorted_symbols is not None:
            return self._sorted_symbols

        all_symbols = self.collect_all_symbols()
        self._sorted_symbols = sorted(list(all_symbols), key=lambda s: str(s))
        return self._sorted_symbols

    def get_symbol_to_index_mapping(self) -> Dict[Symbol, int]:
        """
        获取符号到索引的映射

        Returns:
            Dict[Symbol, int]: 符号到索引的映射字典
        """
        if self._symbol_to_index is not None:
            return self._symbol_to_index

        sorted_symbols = self.get_sorted_symbols()
        self._symbol_to_index = {sym: i for i, sym in enumerate(sorted_symbols)}
        return self._symbol_to_index

    @property
    def all_symbols(self) -> Set[Symbol]:
        """获取所有符号"""
        return self.collect_all_symbols()

    @property
    def sorted_symbols(self) -> List[Symbol]:
        """获取排序后的符号列表"""
        return self.get_sorted_symbols()

    @property
    def symbol_to_index(self) -> Dict[Symbol, int]:
        """获取符号到索引的映射"""
        return self.get_symbol_to_index_mapping()

    @property
    def n_variables(self) -> int:
        """获取变量数量"""
        return len(self.sorted_symbols)
