"""
PDE 评估器（基于 DiffEq.PDE）

最小化实现：
- 支持一维热方程等简单 PDE 的数值仿真（py-pde）并计算终端场的 L2 范数等指标；
- 设计独立、低耦合；由 ParseResult 选择性挂载（当 functional_config 为 PDEConfig 时）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import sympy as sp

from .spec import FunctionalSpec
import os, sys

# 兼容本地源码布局：确保 DiffEq 子包可被导入
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_DIFFEQ = os.path.join(_ROOT, 'DiffEq')
for _p in (_ROOT, _DIFFEQ):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from .config import PDEConfig, MetricSpec


class PDEEvaluator:
    """PDE 轨迹评估器（简化）"""

    def __init__(self, cfg: PDEConfig):
        self.cfg = cfg
        self._compiled = False

    def compile(self) -> None:
        # 占位：未来可在此进行 PDE 标准化与检查
        self._compiled = True

    def simulate(self, opt_vars: Dict[str, float], solver_override: Optional[str] = None, dt_override: Optional[float] = None) -> Any:
        if not self._compiled:
            self.compile()
        # 组织初始场
        grid_spec = self.cfg.grid_spec or {}
        state = self.cfg.init_field
        if callable(state):
            # 仅支持 1D：依据 grid_spec 生成 x 网格
            bounds = grid_spec.get('bounds') or ((0.0, 1.0),)
            shape = grid_spec.get('shape') or (64,)
            xa, xb = float(bounds[0][0]), float(bounds[0][1])
            Nx = int(shape[0])
            x = np.linspace(xa, xb, Nx)
            state = state(x)
        # 常数/参数
        consts = {}
        for s, v in (self.cfg.consts or {}).items():
            consts[s] = float(v)
        for s in getattr(self.cfg, 'param_symbols', []) or []:
            if str(s) in opt_vars:
                consts[s] = float(opt_vars[str(s)])
        # 求解
        from DiffEq.PDE.PDEModule import PDEModule
        pde_mod = PDEModule(self.cfg.pde)
        # 缺省边界条件：若为非周期网格，使用 0-Dirichlet（各轴两端0值），以匹配常见热方程设置
        bc_default = None
        try:
            bounds = grid_spec.get('bounds') or ((0.0, 1.0),)
            periodic = bool(grid_spec.get('periodic', False))
            if not periodic and isinstance(bounds, (list, tuple)):
                ax_labels = ['x', 'y', 'z']
                bc_default = {}
                for i in range(len(bounds)):
                    ax = ax_labels[i] if i < len(ax_labels) else f'a{i}'
                    bc_default[f"{ax}-"] = {"value": 0.0}
                    bc_default[f"{ax}+"] = {"value": 0.0}
        except Exception:
            bc_default = None

        # 允许覆盖求解器与时间步长（用于数值稳定性兜底）
        solver_sel = solver_override if solver_override is not None else "scipy"
        dt_sel = float(dt_override) if dt_override is not None else float(self.cfg.dt)
        sol = pde_mod.num_solve(
            state=state,
            t_range=(float(self.cfg.t0), float(self.cfg.t1)),
            dt=dt_sel,
            const_cond=consts,
            solver=solver_sel,
            bc=bc_default,
            bc_ops=None,
            grid_spec=grid_spec,
        )
        return sol  # py-pde 的 Field/FieldCollection

    def evaluate_all(self, opt_vars: Dict[str, float], metrics: Optional[List[MetricSpec]] = None) -> Dict[str, float]:
        if metrics is None or len(metrics) == 0:
            metrics = []
        sol = self.simulate(opt_vars)
        res: Dict[str, float] = {}
        # 取末时刻场（不同 py-pde 版本兼容）：
        field_final = None
        # Trajectory 支持索引为末帧 Field
        try:
            field_final = sol[-1]
        except Exception:
            field_final = None
        if field_final is None:
            # 若 simulate 直接返回 Field
            field_final = sol
        # 提取数值数组
        try:
            arr = field_final.data if hasattr(field_final, 'data') else field_final.to_numpy()
        except Exception:
            arr = np.asarray(field_final)
        # 数值稳定性兜底：若末帧异常（发散/非有限），尝试缩小 dt 并改用显式格式重算
        def _need_resim(a: np.ndarray) -> bool:
            try:
                if not np.all(np.isfinite(a)):
                    return True
                if np.max(np.abs(a)) > 1e6:
                    return True
                return False
            except Exception:
                return True

        if _need_resim(arr):
            try:
                grid_spec = self.cfg.grid_spec or {}
                bounds = grid_spec.get('bounds') or ((0.0, 1.0),)
                shape = grid_spec.get('shape') or (len(np.asarray(arr).ravel()),)
                xa, xb = float(bounds[0][0]), float(bounds[0][1])
                Nx = int(shape[0]) if shape else max(2, np.asarray(arr).ravel().shape[0])
                dx = (xb - xa) / max(Nx - 1, 1)
                # 保守稳定步长：dt <= 0.45 * dx^2 / k_max；若无法解析 k，取 k_max=1.0
                k_max = 1.0
                dt_new = 0.45 * (dx ** 2) / max(k_max, 1e-12)
                sol = self.simulate(opt_vars, solver_override='explicit', dt_override=dt_new)
                try:
                    field_final = sol[-1]
                    arr = field_final.data if hasattr(field_final, 'data') else field_final.to_numpy()
                except Exception:
                    pass
            except Exception:
                pass

        for m in metrics:
            agg = (m.agg or '').lower()
            if agg in {'pde_final_l2', 'terminal_l2', 'l2'}:
                # 空间积分 L2（1D）：按网格积分近似
                grid_spec = self.cfg.grid_spec or {}
                bounds = grid_spec.get('bounds') or ((0.0, 1.0),)
                shape = grid_spec.get('shape') or (len(arr),)
                xa, xb = float(bounds[0][0]), float(bounds[0][1])
                Nx = int(shape[0]) if shape else arr.shape[0]
                x = np.linspace(xa, xb, Nx)
                val = float(np.sqrt(np.trapz((np.asarray(arr).ravel())**2, x)))
                res[m.id] = val
            elif agg in {'pde_final_l1', 'terminal_l1', 'l1'}:
                res[m.id] = float(np.sum(np.abs(arr)))
            else:
                res[m.id] = float(np.sqrt(np.sum(arr ** 2)))
        return res


def build_pde_evaluator(cfg: PDEConfig, objective_key: Any, meta_override: Dict[str, Any] | None = None) -> FunctionalSpec:
    """构造 PDE evaluator（单指标包装）"""
    ev = PDEEvaluator(cfg)
    # 构造单个指标（若 meta_override 提供 aggregation 标识）
    if meta_override is not None:
        agg = str(meta_override.get('aggregation', 'pde_final_l2'))
        metrics = [MetricSpec(id=f"obj:{objective_key}", kind='path', agg=agg)]
    else:
        metrics = cfg.metrics or [MetricSpec(id=f"obj:{objective_key}", kind='path', agg='pde_final_l2')]

    def evaluator(opt_vars: Dict[str, float]) -> float:
        vals = ev.evaluate_all(opt_vars, metrics)
        return float(vals.get(metrics[0].id, 0.0))

    spec = FunctionalSpec(
        evaluator=evaluator,
        metadata={"kind": "pde", "aggregation": metrics[0].agg},
    )
    return spec

__all__ = ["PDEEvaluator", "build_pde_evaluator"]
