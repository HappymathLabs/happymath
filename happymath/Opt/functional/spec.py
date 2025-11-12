"""
Functional specification and evaluation interfaces.

Notes:
- Defines a unified description for functional (differential/integral/simulation)
  objectives/constraints in Opt.
- By attaching FunctionalSpec to the IR, adapters (pymoo/pyomo) can uniformly
  consume the evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass(slots=True)
class FunctionalSpec:
    """Specification for functional objectives/constraints.

    Fields:
    - evaluator: Callable taking `opt_vars: Dict[str, float]`, returns a scalar or aggregatable array.
    - metadata: Extra metadata (e.g., system_id/domain/aggregation/controls).
    - cache_key: Optional cache key to share a single simulation result across objectives/constraints.
    """

    evaluator: Callable[[Dict[str, float]], Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    cache_key: Optional[str] = None


__all__ = ["FunctionalSpec"]
