"""ODE-to-SciPy adapter entry.

Lightweight dispatch that routes to IVP/BVP adapters; mirrors the PDE adapter design.
"""

from typing import Dict, Any, Optional, Tuple
from .ivp_adapter import build_ivp_adapter
from .bvp_adapter import build_bvp_adapter
from happymath.DiffEq.diffeq_core.de_exceptions import InvalidParameterError
import logging


def ode2scipy(ctx, mode: str, cond: Dict, const_cond: Optional[Dict] = None) -> Tuple:
    """Convert a SymPy ODE into SciPy-compatible callable(s).

    Args:
        ctx: ODE context (ODEModule).
        mode: Solve mode ('IVP' or 'BVP').
        cond: Conditions mapping (initial or boundary conditions).
        const_cond: Optional constants mapping.

    Returns:
        For IVP: (scipy_ode_func, S_values, const_values)
        For BVP: (scipy_ode_func, bc_func, S_values, const_values)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # 验证模式参数
        mode_upper = mode.upper()
        if mode_upper not in ["IVP", "BVP"]:
            raise InvalidParameterError(
                "mode", mode, valid_values=["IVP", "BVP"]
            )
        
        logger.debug(f"Building {mode_upper} adapter")
        
        # 根据模式直接调用对应的适配器
        if mode_upper == "IVP":
            return build_ivp_adapter(ctx, cond, const_cond)
        elif mode_upper == "BVP":
            return build_bvp_adapter(ctx, cond, const_cond)
            
    except Exception as e:
        logger.error(f"Failed to build ODE adapter: {str(e)}")
        raise
