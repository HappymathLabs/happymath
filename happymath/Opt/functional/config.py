"""
FUNCTIONAL 配置结构

说明：
- 该模块为功能型问题提供统一的配置入口，便于 evaluator 构造时读取。
- 采用轻量 dataclass，调用方（OptModule/__init__）可通过 kwargs 传入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal
import sympy as sp


@dataclass(slots=True)
class ControlParamConfig:
    """控制参数化配置

    - kind: 'piecewise_constant' 等
    - func: 控制函数符号，如 u(t)
    - coeff_symbols: 系数符号列表，用作决策变量（需被 VariableManager 纳入）
    - segments: 段数（分段常数）
    - bounds: 系数统一边界 (lb, ub)，用于 BoundManager
    """

    kind: str = "piecewise_constant"
    func: Optional[sp.Function] = None
    coeff_symbols: List[sp.Symbol] = field(default_factory=list)
    segments: int = 10
    bounds: Optional[tuple] = None


@dataclass(slots=True)
class DomainConfig:
    """连续域配置（目前仅支持 1D：时间域）"""

    var: sp.Symbol
    t0: float
    t1: float
    grid_n: int = 101


@dataclass(slots=True)
class WindowSpec:
    """时间窗口规格

    说明：
    - 用于在评估器中裁剪积分/路径/终端指标的时间范围；
    - 若为 None 则表示使用整个域 [t0, t1]。
    """

    t0: float
    t1: float


@dataclass(slots=True)
class IntegrandSpec:
    """被积函数规格

    字段：
    - id: 唯一标识，便于评估结果对齐
    - expr: SymPy 表达式，可包含 t、状态 x(t)、导数 x'(t)、控制 u(t)、参数等；
    - window: 可选时间窗口，None 表示全窗口；
    - channel: 逻辑通道标识（可选），用于多通道功率叠加等场景中的分组。
    """

    id: str
    expr: sp.Expr
    window: Optional[WindowSpec] = None
    channel: Optional[str] = None


@dataclass(slots=True)
class MetricSpec:
    """指标规格

    字段：
    - id: 唯一标识（如 'obj:0'、'con:c1'），便于一次仿真返回多指标时对齐；
    - kind: 指标类型：
        'integral'（积分指标，对应 integrand）
        'terminal'（终端值，需 state_index 或 expr 给出）
        'path'（路径聚合，如 l2_norm/max_abs/mean_abs 等）
    - integrand: 若为 integral 则需提供 IntegrandSpec；
    - state_index: 可选的状态列索引（terminal/path 可用，默认 0）；
    - agg: 聚合方式：
        - 对 integral：可标注 'trapz'/'simpson'（默认trapz）；
        - 对 path：可选 'l2_norm'/'l1_norm'/'max_abs'/'mean_abs' 等；
        - 对 terminal：忽略。
    """

    id: str
    kind: Literal['integral', 'terminal', 'path']
    integrand: Optional[IntegrandSpec] = None
    state_index: Optional[int] = None
    agg: str = 'trapz'


@dataclass(slots=True)
class ODEIVPConfig:
    """ODE/IVP 的功能型配置

    - ode: ODE 方程或方程组（SymPy Eq 列表）
    - domain: 连续域配置
    - ivp_conds: 初值条件字典，如 {x(0): 0, y(0): y0_sym}
    - constants: 常数/系数字典，如 {a: 1.0, b: b_sym}
    - control: 控制参数化配置（可选）
    - objective_meta: 目标聚合元信息（按目标索引或名称组织），如 {0: {"aggregation": "integral", "expr": u(t)**2}}
    - constraint_meta: 约束聚合元信息（可选），如 {"c1": {"aggregation": "final_state", "expr": x(domain.t1)}}
    - extra_symbols: 附加为决策变量引入的符号列表（如初值/常数/控制系数），用于 VariableManager
    - bounds: 针对 extra_symbols 的边界映射 {symbol: (lb, ub)}
    - metrics: 指标清单（可选）。若为空，评估器将基于 objective_meta/constraint_meta 衍生生成默认指标。
    - param_symbols: 额外标量参数（如 C_h、C_rot、比例系数、α），注册为决策变量；
    - param_bounds: 参数边界映射。
    """

    ode: List[sp.Eq]
    domain: DomainConfig
    ivp_conds: Dict[Any, Any] = field(default_factory=dict)
    constants: Dict[sp.Symbol, Any] = field(default_factory=dict)
    control: Optional[ControlParamConfig] = None
    objective_meta: Dict[Any, Dict[str, Any]] = field(default_factory=dict)
    constraint_meta: Dict[Any, Dict[str, Any]] = field(default_factory=dict)
    extra_symbols: List[sp.Symbol] = field(default_factory=list)
    bounds: Dict[sp.Symbol, tuple] = field(default_factory=dict)
    metrics: List[MetricSpec] = field(default_factory=list)
    param_symbols: List[sp.Symbol] = field(default_factory=list)
    param_bounds: Dict[sp.Symbol, tuple] = field(default_factory=dict)


__all__ = [
    "ControlParamConfig",
    "DomainConfig",
    "WindowSpec",
    "IntegrandSpec",
    "MetricSpec",
    "ODEIVPConfig",
]

# === P3: BVP 与 PDE 配置 ===

@dataclass(slots=True)
class ODEBVPConfig:
    """ODE/BVP 的功能型配置（简化版）

    - ode: ODE 方程或方程组（SymPy Eq 列表）
    - domain: 连续域配置
    - bvp_conds: 边界条件字典（简化支持 x(t0)=v 与 x(t1)=v 形式），例如 {x(t0): 0.0, x(t1): 1.0}
    - constants: 常数/系数字典
    - control: 控制参数化（可选，当前 BVP 评估器不使用控制）
    - metrics/param_symbols/param_bounds：与 IVP 配置一致
    """

    ode: List[sp.Eq]
    domain: DomainConfig
    bvp_conds: Dict[Any, Any] = field(default_factory=dict)
    constants: Dict[sp.Symbol, Any] = field(default_factory=dict)
    control: Optional[ControlParamConfig] = None
    metrics: List[MetricSpec] = field(default_factory=list)
    param_symbols: List[sp.Symbol] = field(default_factory=list)
    param_bounds: Dict[sp.Symbol, tuple] = field(default_factory=dict)


@dataclass(slots=True)
class PDEConfig:
    """PDE 配置（最小化支持）

    - pde: PDE 表达式或表达式组（SymPy Eq 列表）
    - t0, t1, dt: 时间范围与步长
    - grid_spec: 网格规格（当 state 为 numpy 时）：{"bounds": ((xa, xb),), "shape": (Nx,), "periodic": False}
    - init_field: 初始场构造，支持：
        * numpy.ndarray（将配合 grid_spec 构场）
        * callable(x_grid) -> numpy.ndarray
    - metrics: 仅支持终端场的 L2 范数等（由 evaluator 解析）
    - consts: PDE 中的常数参数（可含优化变量 param_symbols）
    - param_symbols/param_bounds：参数变量定义
    """

    pde: List[sp.Eq]
    t0: float
    t1: float
    dt: float
    grid_spec: Dict[str, Any]
    init_field: Any
    metrics: List[MetricSpec] = field(default_factory=list)
    consts: Dict[sp.Symbol, Any] = field(default_factory=dict)
    param_symbols: List[sp.Symbol] = field(default_factory=list)
    param_bounds: Dict[sp.Symbol, tuple] = field(default_factory=dict)

__all__.extend(["ODEBVPConfig", "PDEConfig"])
