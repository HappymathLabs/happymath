"""
分析器抽象基类

定义所有表达式分析器的标准接口。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class AnalyzerBase(ABC):
    """表达式分析器抽象基类"""

    def __init__(self, sympy_obj, value_range='real', **kwargs):
        """
        初始化分析器

        Args:
            sympy_obj: SymPy表达式或表达式集合
            value_range: 变量取值范围 ('real', 'complex', 'integer'等)
            **kwargs: 其他参数
        """
        self.sympy_obj = sympy_obj
        self.value_range = value_range
        self._analysis_cache = {}  # 缓存分析结果
        self._analyzed = False

    @abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """
        执行分析，返回分析结果字典

        Returns:
            Dict[str, Any]: 分析结果字典
        """
        pass

    @abstractmethod
    def get_symbols(self):
        """
        获取所有符号变量

        Returns:
            set: 符号变量集合
        """
        pass

    def _cache_result(self, key: str, value: Any):
        """缓存分析结果"""
        self._analysis_cache[key] = value

    def _get_cached_result(self, key: str, default=None):
        """获取缓存的分析结果"""
        return self._analysis_cache.get(key, default)

    def _clear_cache(self):
        """清空缓存"""
        self._analysis_cache.clear()
        self._analyzed = False
