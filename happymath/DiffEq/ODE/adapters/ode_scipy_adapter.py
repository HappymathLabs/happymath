"""ODE到SciPy适配器入口模块

简化的ODE适配器入口，参考PDE的简洁设计。
将复杂的工厂模式简化为直接的函数调用路由。
"""

from typing import Dict, Any, Optional, Tuple
from .ivp_adapter import build_ivp_adapter
from .bvp_adapter import build_bvp_adapter
from happymath.DiffEq.diffeq_core.de_exceptions import InvalidParameterError
import logging


def ode2scipy(ctx, mode: str, cond: Dict, const_cond: Optional[Dict] = None) -> Tuple:
    """
    将sympy格式的ODE转换为scipy标准格式的主入口函数
    
    简化的设计，直接根据模式调用对应的适配器函数。
    保持与原始接口完全兼容，确保现有测试可以正常通过。
    
    Args:
        ctx: ODE上下文对象（ODEModule实例）
        mode: 求解模式 ('IVP' 或 'BVP')
        cond: 条件字典，包含初值条件或边界条件
        const_cond: 常数条件字典（可选）
        
    Returns:
        根据模式返回不同的元组：
        - IVP模式: (scipy_ode_func, S_values, const_values)
          - scipy_ode_func: scipy兼容的ODE函数
          - S_values: 初值列表
          - const_values: 常数值列表
          
        - BVP模式: (scipy_ode_func, bc_func, S_values, const_values)
          - scipy_ode_func: scipy兼容的ODE函数
          - bc_func: 边界条件函数
          - S_values: 初值列表
          - const_values: 常数值列表
    
    Raises:
        InvalidParameterError: 参数无效
        ConditionValidationError: 条件验证失败
        BVPValidationError: BVP条件验证失败（仅BVP模式）
    
    Examples:
        >>> # IVP求解
        >>> ode_func, initial_values, constants = ode2scipy(ode_ctx, "IVP", ivp_conditions, const_dict)
        
        >>> # BVP求解
        >>> ode_func, bc_func, initial_values, constants = ode2scipy(ode_ctx, "BVP", bvp_conditions, const_dict)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # 验证模式参数
        mode_upper = mode.upper()
        if mode_upper not in ["IVP", "BVP"]:
            raise InvalidParameterError(
                "mode", mode, valid_values=["IVP", "BVP"]
            )
        
        logger.debug(f"构建{mode_upper}适配器")
        
        # 根据模式直接调用对应的适配器
        if mode_upper == "IVP":
            return build_ivp_adapter(ctx, cond, const_cond)
        elif mode_upper == "BVP":
            return build_bvp_adapter(ctx, cond, const_cond)
            
    except Exception as e:
        logger.error(f"ODE适配器构建失败: {str(e)}")
        raise
