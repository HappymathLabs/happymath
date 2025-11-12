"""
FUNCTIONAL 规格与评估接口定义

说明：
- 本文件定义功能型（微分/积分/仿真）目标/约束在 Opt 中的统一描述。
- 通过在 IR 中挂载 FunctionalSpec，使得适配器（pymoo/pyomo）可以统一消费 evaluator。

风格约定：中文注释；尽量保持高内聚低耦合，避免侵入其他模块。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass(slots=True)
class FunctionalSpec:
    """功能型规格

    字段：
    - evaluator: 可调用对象，入参为 `opt_vars: Dict[str, float]`，返回标量或可聚合的数组
    - metadata: 附加元数据（如：system_id/domain/aggregation/controls 配置等）
    - cache_key: 可选缓存键（用于跨目标/约束共享一次仿真结果）
    """

    evaluator: Callable[[Dict[str, float]], Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    cache_key: Optional[str] = None


__all__ = ["FunctionalSpec"]

