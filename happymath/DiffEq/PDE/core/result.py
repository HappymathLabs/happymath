from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PDESolutionResult:
    """PDE 求解结果的数据类"""
    solution: Any
    time_range: Any
    dt: float
    solver: str
    constants: Dict[str, Any]
    rhs: Dict[str, Any]
    success: bool
    message: str = ""


