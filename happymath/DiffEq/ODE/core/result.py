from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Union
import numpy as np


@dataclass
class ODESolutionResult:
    """Container for ODE numeric solution results.

    Attributes:
        domain: Sample points of the independent variable (e.g., time grid).
        solution: Solution values at `domain`, shaped (n_points, n_states).
        error: Local error estimates or a constant placeholder list.
        solution_func: Optional callable f(t) that evaluates the continuous solution if available.
        substitution_dict: Mapping of inputs used for solve (e.g., initial conditions, constants).
        success: Whether the solver reports success.
        message: Human-readable status message.
    """
    domain: np.ndarray
    solution: np.ndarray
    error: Union[np.ndarray, List[float]]
    solution_func: Callable
    substitution_dict: Dict[Any, Any]
    success: bool
    message: str = ""

