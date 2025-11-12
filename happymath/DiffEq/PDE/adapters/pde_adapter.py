"""
PDE adapter module.

Convert standardized expressions from the expression parser into py-pde solvable objects and execute.
"""

from typing import Any, Dict, Optional, Tuple, Sequence
from ...diffeq_core.de_exceptions import MissingParameterError
import numpy as np
import pde
from ..core import PDESolutionResult


def solve_pde(ctx,
              state: Any,
              t_range: Any,
              dt: float,
              solver: str = "explicit",
              const_cond: Optional[Dict] = None,
              bc: Optional[dict | str] = None,
              bc_ops: Optional[Dict] = None,
              grid_spec: Optional[Dict] = None):
    """
    Build a py-pde PDE object from the context's standardized result and solve.

    Args:
        ctx: PDEModule instance.
        state: Initial field (py-pde Field/FieldCollection or numpy array/dict of arrays).
        t_range: Time range.
        dt: Time step.
        solver: Solver name forwarded to py-pde.
        const_cond: Constants dictionary.

    Returns:
        PDESolutionResult
    """
    solvable_pde_dict = ctx.to_solvable_pde

    const_dict = {}
    if ctx.free_consts:
        if const_cond is None:
            raise MissingParameterError("const_cond", context="PDE constants are required")
        const_dict = dict(const_cond)

    # 1) Prepare initial field: allow numpy / dict[str, np.ndarray] / Field
    prepared_state = _prepare_state(state, grid_spec)

    # 2) Normalize constants: scalars unchanged; ndarray -> Field; vector (len=dim) -> VectorField
    consts_ready = _prepare_consts(const_dict, prepared_state)

    # 3) Choose RHS representation: string or callable
    rhs = _maybe_rewrite_rhs_strings(solvable_pde_dict, prepared_state)
    if rhs is None:
        # Fall back to function form (e.g., includes first-order derivatives/complex terms)
        rhs = _make_function_rhs(ctx, prepared_state, consts_ready, bc)
    else:
        # String RHS: auto-inject basis vectors ex/ey/ez when referenced
        _inject_basis_vectors(consts_ready, rhs, prepared_state)

    # 4) Build and solve
    pde_obj = pde.PDE(rhs=rhs, consts=consts_ready, bc=bc, bc_ops=bc_ops)
    raw_solution = pde_obj.solve(prepared_state, t_range, dt, solver=solver)

    return PDESolutionResult(
        solution=raw_solution,
        time_range=t_range,
        dt=dt,
        solver=solver,
        constants=const_dict,
        rhs=rhs,
        success=True,
        message="PDE solve succeeded"
    )


def _prepare_state(state: Any, grid_spec: Optional[Dict]) -> pde.FieldBase:
    """Normalize state into a py-pde Field/FieldCollection.

    Supports:
    - Already FieldBase / FieldCollection: return as-is;
    - numpy.ndarray: build CartesianGrid + ScalarField via grid_spec;
    - dict[str, np.ndarray]: multi-field, build FieldCollection in key order.
    """
    if hasattr(state, 'grid') and hasattr(state, 'data'):
        # 已是 FieldBase 或 FieldCollection
        return state

    if grid_spec is None:
        grid_spec = {}

    bounds: Optional[Sequence[Tuple[float, float]]] = grid_spec.get('bounds')
    shape: Optional[Sequence[int]] = grid_spec.get('shape')
    periodic = grid_spec.get('periodic', False)

    def _mk_grid_from_shape(_shape):
        if bounds is None:
            # Default each axis to [0, 1]
            _bounds = tuple((0.0, 1.0) for _ in range(len(_shape)))
        else:
            _bounds = tuple(bounds)
        return pde.CartesianGrid.from_bounds(_bounds, tuple(_shape), periodic=periodic)

    if isinstance(state, np.ndarray):
        arr = np.asarray(state)
        if shape is None:
            shape = arr.shape
        grid = _mk_grid_from_shape(shape)
        return pde.ScalarField(grid, data=arr)

    if isinstance(state, dict):
        # Multi-field: require identical array shapes
        keys = list(state.keys())
        arr0 = np.asarray(state[keys[0]])
        if shape is None:
            shape = arr0.shape
        grid = _mk_grid_from_shape(shape)
        fields = []
        for k in keys:
            fields.append(pde.ScalarField(grid, data=np.asarray(state[k])))
        return pde.FieldCollection(fields, labels=keys)

    raise TypeError("Unsupported state type. Please pass py-pde Field, numpy.ndarray, or dict[str, ndarray].")


def _prepare_consts(consts: Dict[str, Any], state_field: pde.FieldBase) -> Dict[str, Any]:
    """Prepare constants/coefs:
    - Scalars unchanged;
    - ndarray matching state shape -> ScalarField;
    - 1D vector (len=dim) -> constant VectorField;
    - Existing Field -> pass-through.
    """
    if not consts:
        return {}

    grid = state_field.grid if hasattr(state_field, 'grid') else state_field[0].grid
    dim = grid.num_axes
    out: Dict[str, Any] = {}
    for name, val in consts.items():
        if hasattr(val, 'grid') and hasattr(val, 'data'):
            out[name] = val
            continue
        if np.isscalar(val):
            out[name] = float(val)
            continue
        arr = np.asarray(val)
        if arr.ndim == 1 and arr.size == dim:
            # 方向常量向量，例如平流速度
            out[name] = pde.VectorField(grid, data=arr)
        else:
            # 空间变系数
            out[name] = pde.ScalarField(grid, data=arr)
    return out


def _maybe_rewrite_rhs_strings(rhs_map: Dict[str, str], state_field: pde.FieldBase) -> Optional[Dict[str, str]]:
    """Pass solvable_format as string directly to py-pde when possible.

    Notes:
    - py-pde natively supports operators like d_dx/d_dy/d2_dx2/d2_dy2/laplace/gradient/dot;
      no need to fall back to callables.
    - In 1D, optionally rewrite d2_dx2 to laplace for compactness (non-essential).
    """
    if not rhs_map:
        return None
    grid = state_field.grid if hasattr(state_field, 'grid') else state_field[0].grid
    dim = grid.num_axes
    out: Dict[str, str] = {}
    for k, expr in rhs_map.items():
        new_expr = expr
        # 可选：1D 时将 d2_dx2 写成 laplace（非必须，仅微调）
        if dim == 1:
            new_expr = new_expr.replace('d2_dx2', 'laplace')
        out[k] = new_expr
    return out


def _make_function_rhs(ctx, state_field: pde.FieldBase, consts: Dict[str, Any], bc: Any):
    """Build a function-based RHS for PDEs with first-order or complex terms.

    Current version:
    - Supports single scalar field or FieldCollection (component-wise);
    - Uses key order from ctx.to_solvable_pde to pick state components;
    - Supports common terms: laplace, gradient, dot(v, gradient(u)) where v can be constant VectorField.
    """
    # 从 ctx 得到字段顺序
    rhs_dict = ctx.to_solvable_pde
    keys = list(rhs_dict.keys())

    def rhs_callable(state, t=0.0):
        # 将 state 统一成 list[ScalarField]
        if hasattr(state, 'fields'):  # FieldCollection
            fields = list(state.fields)
        else:
            fields = [state]
        out_fields = []
        for i, key in enumerate(keys):
            u = fields[i]
            # 尝试从原始表达式文本判断是否包含一阶导：若包含 d_dx -> 用 gradient 近似
            expr_text = rhs_dict[key]
            # 支持平流：v * d_dx(u) ≈ dot(v, grad(u))
            grad_u = u.gradient(bc=bc)
            evo = None
            if 'd2_dx2' in expr_text or 'laplace' in expr_text:
                evo = u.laplace(bc=bc)
            if 'd_dx' in expr_text:
                # 需要常量向量 v（若 consts 中存在标量 v，则尝试扩成向量）
                # 查找可能的速度常量名列表（简单启发式）
                cand = None
                for name in consts.keys():
                    if name.lower() in ('v', 'vel', 'velocity', 'c'):
                        cand = name
                        break
                if cand is not None:
                    v = consts[cand]
                    if np.isscalar(v):
                        grid = u.grid
                        v = pde.VectorField(grid, data=np.array([float(v)] + [0.0]*(grid.num_axes-1)))
                    if evo is None:
                        evo = v.dot(grad_u)
                    else:
                        evo = evo + v.dot(grad_u)
                else:
                    # 若没有速度，退化为 x 向分量
                    comp0 = grad_u[0]
                    evo = comp0 if evo is None else (evo + comp0)

            # 缺省回退
            if evo is None:
                evo = u.laplace(bc=bc)  # 最保守的回退
            out_fields.append(evo)

        # 输出与 state 同结构
        if hasattr(state, 'fields'):
            return pde.FieldCollection(out_fields, labels=keys)
        return out_fields[0]

    return rhs_callable


def _inject_basis_vectors(consts: Dict[str, Any], rhs_map: Dict[str, str], state_field: pde.FieldBase) -> None:
    """Auto-inject ex/ey/ez basis vectors into consts when referenced in string RHS."""
    grid = state_field.grid if hasattr(state_field, 'grid') else state_field[0].grid
    dim = grid.num_axes
    text = "\n".join(rhs_map.values())

    def ensure_vec(name: str, comp_index: int):
        if name in consts:
            return
        vec = np.zeros(dim, dtype=float)
        if comp_index < dim:
            vec[comp_index] = 1.0
        consts[name] = pde.VectorField(grid, data=vec)

    if 'ex' in text:
        ensure_vec('ex', 0)
    if 'ey' in text and dim >= 2:
        ensure_vec('ey', 1)
    if 'ez' in text and dim >= 3:
        ensure_vec('ez', 2)
