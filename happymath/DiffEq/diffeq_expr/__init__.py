"""Module exports.
SymPy前端包 - 微分方程表达式处理器
从ODEModule和PDEModule中抽离的符号处理前端包
"""

from .processor import (
    ExpressionProcessor,
    process_expression,
    analyze_expression,
    validate_expression
)
from .base import (
    AbstractExpressionAnalyzer,
    AbstractSymbolManager,
    AbstractStandardizer,
    AbstractResult
)
from .DE import (
    ODEAnalyzer,
    PDEAnalyzer,
    ODESymbolManager,
    PDESymbolManager,
    ODEStandardizer,
    PDEStandardizer,
    ODEResult,
    PDEResult
)

# Removed hardcoded version - use happymath.__version__ instead

__all__ = [
    "ExpressionProcessor",
    "process_expression",
    "analyze_expression",
    "validate_expression",
    "AbstractExpressionAnalyzer",
    "AbstractSymbolManager", 
    "AbstractStandardizer",
    "AbstractResult",
    "ODEAnalyzer",
    "PDEAnalyzer",
    "ODESymbolManager",
    "PDESymbolManager",
    "ODEStandardizer",
    "PDEStandardizer",
    "ODEResult",
    "PDEResult"
]