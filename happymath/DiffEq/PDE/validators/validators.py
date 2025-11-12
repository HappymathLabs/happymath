"""
PDE validators 模块

提供基础的条件与参数校验工具，保持与 ODE 目录对齐的结构。
"""

from collections.abc import Iterable
from typing import Any


class PDEParameterValidator:
    @staticmethod
    def validate_time_range(t_range: Any) -> None:
        if not isinstance(t_range, Iterable) and not (isinstance(t_range, tuple) and len(t_range) == 2):
            raise TypeError("'t_range' must be an iterable or a (t0, t1) tuple.")

    @staticmethod
    def validate_time_step(dt: Any) -> None:
        try:
            dt_val = float(dt)
        except Exception:
            raise TypeError("'dt' must be a number.")
        if dt_val <= 0:
            raise ValueError("'dt' must be positive.")


