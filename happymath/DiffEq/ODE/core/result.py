from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Union
import numpy as np


@dataclass
class ODESolutionResult:
    """ODE求解结果的数据类"""
    domain: np.ndarray
    solution: np.ndarray
    error: Union[np.ndarray, List[float]]
    solution_func: Callable
    substitution_dict: Dict[Any, Any]
    success: bool
    message: str = ""


