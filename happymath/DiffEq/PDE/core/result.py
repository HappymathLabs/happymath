from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PDESolutionResult:
    """Container for PDE numeric solution results.

    Attributes:
        solution: Backend-specific solution object (e.g., py-pde Trajectory).
        time_range: Time range used for integration.
        dt: Time step size used by the solver.
        solver: Solver identifier passed to the backend.
        constants: Constants map supplied to the PDE.
        rhs: Right-hand side representation (callable or expression strings).
        success: Whether the solver reports success.
        message: Human-readable status message.
    """
    solution: Any
    time_range: Any
    dt: float
    solver: str
    constants: Dict[str, Any]
    rhs: Dict[str, Any]
    success: bool
    message: str = ""

