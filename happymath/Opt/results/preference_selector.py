"""
Pymoo结果参考点/理想点ASF择优模块

该模块在OptModule.solve阶段对Pymoo前沿结果执行后处理：
- ref为None时使用无先验理想点ASF（等权）；
- ref为字典时解析SymPy目标表达式对应的参考点，执行参考点ASF；
- 仅对成功的Pymoo结果（且包含多行前沿数据）生效；
- 选中解后会同步更新X/F/variables/objective_value，并记录原始前沿供展示。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import sympy as sp

from ..ir.definitions import IROptProblem, IROptVarType

_EPS = 1e-12
_RHO = 1e-6


def select_preferred_from_pymoo(
    results: List[Dict[str, Any]],
    senses: Iterable[str],
    objective_exprs: Iterable[sp.Expr],
    ir_problem: Optional[IROptProblem],
    sorted_symbols: Optional[Iterable[sp.Symbol]],
    ref: Optional[Dict[Any, float]] = None,
) -> List[Dict[str, Any]]:
    """
    对Pymoo结果执行ASF择优，返回修改后的results引用。
    """
    if not results:
        return results

    senses = list(senses or [])
    n_obj = len(senses)
    if n_obj == 0:
        return results

    has_pymoo = any(r.get('solver_type') == 'pymoo' for r in results)
    if not has_pymoo:
        return results

    sense_vec = _build_sense_vector(senses)
    objective_exprs = list(objective_exprs or [])
    if len(objective_exprs) != n_obj:
        # 方向与目标数不匹配时直接跳过，避免潜在错误
        return results

    ref_vector = None
    if ref is not None:
        ref_vector = _parse_reference_dict(ref, objective_exprs)

    symbols = list(sorted_symbols or [])
    if not symbols and ir_problem is not None:
        symbols = [var.symbol for var in ir_problem.variables]

    for result in results:
        if result.get('solver_type') != 'pymoo':
            continue
        if not result.get('success', False):
            continue
        processed = _process_single_result(result, sense_vec, symbols, ir_problem, ref_vector)
        if processed:
            result.update(processed)

    return results


def _process_single_result(
    result: Dict[str, Any],
    sense_vec: np.ndarray,
    sorted_symbols: List[sp.Symbol],
    ir_problem: Optional[IROptProblem],
    ref_vector: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:
    """处理单个Pymoo结果，返回需要更新的字段。"""
    F_raw = result.get('F')
    if F_raw is None:
        return None
    try:
        F = np.asarray(F_raw, dtype=float)
    except Exception:
        return None
    if F.ndim == 1:
        # 只有一行时无需择优
        return None
    if F.shape[0] <= 1 or F.shape[1] != sense_vec.size:
        return None

    X_raw = result.get('X')
    X = None
    if X_raw is not None:
        try:
            X = np.asarray(X_raw, dtype=float)
        except Exception:
            X = None

    n_points = F.shape[0]
    if X is not None:
        if X.ndim == 1:
            if X.size == n_points:
                X = X.reshape(n_points, 1)
            else:
                X = X.reshape(1, X.size)
        if X.shape[0] != n_points:
            # 无法与前沿行数对应时放弃变量更新，只更新F
            X = None

    best_idx, stats = _compute_asf_index(F, sense_vec, ref_vector)
    if best_idx is None:
        return None

    updated: Dict[str, Any] = {}
    selected_F = np.array(F[best_idx], copy=True)
    updated['F'] = selected_F
    updated['objective_value'] = _format_objective_value(selected_F)

    selected_X = None
    if X is not None:
        selected_X = np.array(X[best_idx], copy=True)
        updated['X'] = selected_X
        decoded_vars = _decode_variables(selected_X, ir_problem, sorted_symbols)
        updated['variables'] = decoded_vars

    preference_meta = {
        'mode': 'auto_ideal' if ref_vector is None else 'reference',
        'selected_index': int(best_idx),
        'rho': _RHO,
        'front_size': int(n_points),
        'z_star': stats.z_star.tolist(),
        'z_nadir': stats.z_nadir.tolist(),
    }
    if ref_vector is not None:
        preference_meta['reference'] = [float(v) for v in ref_vector]
    updated.setdefault('postprocess', {})
    postprocess = dict(result.get('postprocess') or {})
    postprocess['preference'] = preference_meta
    postprocess['raw_front'] = np.array(F, copy=True)
    if X is not None:
        postprocess['raw_decisions'] = np.array(X, copy=True)
    updated['postprocess'] = postprocess

    return updated


@dataclass
class _ASFStats:
    z_star: np.ndarray
    z_nadir: np.ndarray
    weights: np.ndarray
    r_hat: np.ndarray


def _compute_asf_index(
    F: np.ndarray,
    sense_vec: np.ndarray,
    ref_vector: Optional[np.ndarray],
) -> tuple[Optional[int], _ASFStats]:
    """执行加权Tchebycheff/ASF并返回最佳索引。"""
    G = F * sense_vec  # 方向统一：越小越好
    z_star = G.min(axis=0)
    z_nadir = G.max(axis=0)
    span = np.maximum(z_nadir - z_star, _EPS)
    Gn = (G - z_star) / span

    weights = np.ones(G.shape[1], dtype=float) / G.shape[1]
    if ref_vector is None:
        r_hat = np.zeros(G.shape[1], dtype=float)
    else:
        g_ref = np.asarray(ref_vector, dtype=float) * sense_vec
        r_hat = (g_ref - z_star) / span

    delta = np.abs(Gn - r_hat)
    penalty = weights * delta
    max_term = np.max(penalty, axis=1)
    sum_term = np.sum(penalty, axis=1)
    scores = max_term + _RHO * sum_term
    if not np.all(np.isfinite(scores)):
        finite_idx = np.where(np.isfinite(scores))[0]
        if finite_idx.size == 0:
            return None, _ASFStats(z_star, z_nadir, weights, r_hat)
        scores = scores.copy()
        scores[~np.isfinite(scores)] = np.inf

    best_idx = int(np.argmin(scores))
    return best_idx, _ASFStats(z_star, z_nadir, weights, r_hat)


def _parse_reference_dict(ref: Dict[Any, float], objective_exprs: List[sp.Expr]) -> np.ndarray:
    """解析参考点字典并返回与目标对应的向量。"""
    if not isinstance(ref, dict):
        raise ValueError("ref参数必须是字典或None。")
    n_obj = len(objective_exprs)
    if len(ref) != n_obj:
        raise ValueError(f"ref字典必须覆盖所有{n_obj}个目标函数。")

    ref_vec = np.zeros(n_obj, dtype=float)
    matched = set()

    for key, value in ref.items():
        idx = _match_objective_index(key, objective_exprs)
        if idx is None:
            raise ValueError(f"无法识别参考点键: {key}")
        if idx in matched:
            raise ValueError(f"参考点键 {key} 存在歧义，请调整表达式。")
        try:
            ref_vec[idx] = float(value)
        except Exception as exc:
            raise ValueError(f"参考点数值 {value} 无法转换为float。") from exc
        matched.add(idx)

    if len(matched) != n_obj:
        raise ValueError("参考点未覆盖全部目标函数。")

    return ref_vec


def _match_objective_index(key: Any, objective_exprs: List[sp.Expr]) -> Optional[int]:
    """根据SymPy表达式或字符串定位目标索引。"""
    for idx, expr in enumerate(objective_exprs):
        if key is expr:
            return idx

    if isinstance(key, sp.Expr):
        for idx, expr in enumerate(objective_exprs):
            if _expr_equals(key, expr):
                return idx
        return None

    # 字符串输入：先直接匹配，再尝试解析为表达式
    if isinstance(key, str):
        for idx, expr in enumerate(objective_exprs):
            if key == str(expr):
                return idx
        try:
            parsed = sp.sympify(key)
        except Exception:
            return None
        for idx, expr in enumerate(objective_exprs):
            if _expr_equals(parsed, expr):
                return idx
    return None


def _expr_equals(a: sp.Expr, b: sp.Expr) -> bool:
    """稳健判断两个表达式是否等价。"""
    if a is b:
        return True
    try:
        if a == b:
            return True
    except Exception:
        pass
    for left, right in ((a, b), (b, a)):
        try:
            if hasattr(left, "equals") and left.equals(right):
                return True
        except Exception:
            continue
    try:
        diff = sp.simplify(a - b)
        return diff == 0
    except Exception:
        return False


def _decode_variables(
    x_vec: Optional[np.ndarray],
    ir_problem: Optional[IROptProblem],
    sorted_symbols: List[sp.Symbol],
) -> Dict[str, Any]:
    """按IR变量定义解码连续/离散取值。"""
    if x_vec is None:
        return {}
    arr = np.asarray(x_vec, dtype=float).flatten()
    symbol_seq = sorted_symbols or []
    if not symbol_seq and ir_problem is not None:
        symbol_seq = [var.symbol for var in ir_problem.variables]
    if not symbol_seq:
        return {}

    var_lookup = {}
    if ir_problem is not None:
        var_lookup = ir_problem.symbol_to_variable()

    decoded: Dict[str, Any] = {}
    for idx, sym in enumerate(symbol_seq):
        if idx >= arr.size:
            break
        var_def = var_lookup.get(sym) if var_lookup else None
        decoded[str(sym)] = _decode_value(arr[idx], var_def)
    return decoded


def _decode_value(value: float, var_def: Optional[Any]) -> Any:
    """根据变量类型执行取整或枚举映射。"""
    if var_def is None:
        return float(value)
    vtype = getattr(var_def, 'var_type', None)
    if vtype == IROptVarType.BINARY:
        return float(np.clip(np.round(value), 0.0, 1.0))
    if vtype == IROptVarType.INTEGER:
        return float(np.round(value))
    if vtype == IROptVarType.ENUM and getattr(var_def, 'discrete_domain', None):
        domain = var_def.discrete_domain.values
        if not domain:
            return float(value)
        try:
            numeric = np.asarray([float(v) for v in domain], dtype=float)
            idx = int(np.argmin(np.abs(numeric - float(value))))
            return domain[idx]
        except Exception:
            idx = int(np.clip(int(np.round(value)), 0, len(domain) - 1))
            return domain[idx]
    return float(value)


def _build_sense_vector(senses: List[str]) -> np.ndarray:
    """将优化方向转换为+1/-1向量。"""
    vec = []
    for sense in senses:
        if isinstance(sense, str) and sense.lower() == 'min':
            vec.append(1.0)
        elif isinstance(sense, str) and sense.lower() == 'max':
            vec.append(-1.0)
        else:
            raise ValueError(f"无法识别的目标方向: {sense}")
    return np.asarray(vec, dtype=float)


def _format_objective_value(f_vec: np.ndarray) -> Any:
    """将目标值格式化为标量或列表。"""
    if f_vec.ndim == 0:
        return float(f_vec)
    flattened = f_vec.flatten()
    if flattened.size == 1:
        return float(flattened[0])
    return [float(v) for v in flattened]
