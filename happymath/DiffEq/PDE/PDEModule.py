import warnings
import sympy
from sympy import Function, solve
from sympy.utilities.iterables import iterable
from ..diffeq_core.de_exceptions import InvalidExpressionError, ExpressionStandardizationError
from ..diffeq_core.de_base import DEBase
from typing import Optional, Dict, List, Union, Any
from ..diffeq_expr import process_expression
from .adapters import solve_pde
from .core import PDESolutionResult

class PDEModule(DEBase):
    def __init__(self, sympy_obj, value_range:str="real", spatial_var_order=["x","y"]):
        """
            【暂不支持】
              1. 暂不支持大于2维空间的偏微分方程；
              2. 暂不支持包含混合偏导数，例如∂²u/∂x∂t，的偏微分方程；
        """
        
        super(PDEModule, self).__init__(sympy_obj, value_range)
        # 基类已持有 _sympy_obj；避免写入只读 property
        self.range = self._value_range
        self.spatial_var_order = spatial_var_order

        if not self.is_pde:
            raise TypeError("This is not an PDE expression, or this is not a standard PDE expression.")

        # 缓存（ExprParser结果）
        self._cached_standard_result: Optional[Any] = None
        self._cache_invalid = True
        
        if not self._check_core_symbol(symbol_str="t"):
            raise InvalidExpressionError(sympy_obj, "The PDE must have time variable 't'.")
        
        # 定义一个空间变量字典，根据list顺序分别存储x,y,z,...这些空间变量
        self.spatial_var_list = []
        self.time_var = None
        for var in self.core_symbol:
            if str(var) != "t":
                if str(var) != "x" and str(var) != "y":
                    warnings.warn(f"{str(var)} is not a standard spatial variable, it will be mapped to x, y or z axis based on the order of acquisition. \
                        You can use 'spatial_var_order' to specify the order of spatial variables.")
                self.spatial_var_list.append(str(var))
            else:
                self.time_var = var
        
        # 根据spatial_var_order对self.spatial_var_list进行排序
        self.spatial_order_var_list = [var for var in self.spatial_var_order if var in self.spatial_var_list]
        if len(self.spatial_order_var_list) == len(self.spatial_var_list):
            self.spatial_var_list = self.spatial_order_var_list
        else:
            warnings.warn(f"The spatial variables in the expression do not match the specified order. Will use the order {self.spatial_var_list} as spatial variables.")

    # 检查是否包含指定核心符号（如时间变量 't'）
    def _check_core_symbol(self, symbol_str: str = "t") -> bool:
        try:
            for symbol in self.core_symbol:
                if str(symbol) == symbol_str:
                    return True
            return False
        except Exception:
            return False

    # 缓存失效
    def _invalidate_cache(self) -> None:
        self._cache_invalid = True

    # 提供与 ODEModule 一致的 expr 属性以触发缓存失效
    @property
    def expr(self) -> Union[sympy.Expr, list]:
        return self._sympy_obj

    @expr.setter
    def expr(self, new_expr: Union[sympy.Expr, list]):
        self._sympy_obj = new_expr
        self._invalidate_cache()
                       
    # ExprParser: 统一标准化入口
    def _compute_standard_pde(self):
        try:
            result = process_expression(self.expr, spatial_var_order=self.spatial_var_order)
            if getattr(result, '_analyzer_result', None) and getattr(result._analyzer_result, 'expression_type', '') != 'PDE':
                raise InvalidExpressionError(self.expr, "This is not a PDE expression.")

            return result
        except Exception as e:
            raise ExpressionStandardizationError(self.expr, "PDE标准化", str(e))

    # 将PDE进行标准化（与旧接口保持一致）
    @property
    def stand_pde(self):
        if self._cache_invalid or self._cached_standard_result is None:
            self._cached_standard_result = self._compute_standard_pde()
            self._cache_invalid = False
        return self._cached_standard_result.standardized_expressions
    
    # 将标准化后的PDE转换为可求解的形式（ExprParser提供）
    @property
    def to_solvable_pde(self):
        if self._cache_invalid or self._cached_standard_result is None:
            _ = self.stand_pde  # 触发缓存
        return self._cached_standard_result.get_solvable_format() if hasattr(self._cached_standard_result, 'get_solvable_format') else getattr(self._cached_standard_result, 'solvable_format', {})
    
    def ana_solve(self):
        """
            求解偏微分方程的解析解
        """
        pass
    
    def num_solve(self, 
                  state,
                  t_range,
                  dt,
                  const_cond: dict = None,
                  solver: str = "explicit",
                  bc: dict | str | None = None,
                  bc_ops: dict | None = None,
                  grid_spec: dict | None = None):
        """
            求解偏微分方程的数值解（外观层接口）

            参数：
            - state: 初始场。可为 py-pde 的 Field/FieldCollection，或 numpy 数组（单场），
                     或 {name: np.ndarray}（多场）。若为数组，将使用 grid_spec 自动构建网格与场。
            - t_range: 时间范围（如 (0, 1) 或 1.0）
            - dt: 时间步长
            - const_cond: 常数/系数字典，支持标量、numpy 数组（将自动转为 Field）、或已有 Field
            - solver: 求解器类型，转发至 py-pde（如 "explicit" 等）
            - bc: 边界条件（py-pde 兼容的 BoundariesData，如 {"x-": {"value": 0}, ...} 或 "periodic"）
            - bc_ops: 针对表达式中各算子的专属边界条件（py-pde 的 bc_ops）
            - grid_spec: 当 state 为 numpy 时，用于构网格的规格字典，支持键：
                         {"bounds": ((xa, xb), ...), "shape": (Nx, ...), "periodic": False}

            返回：
            - py-pde 的解对象（Trajectory）
        """
        detailed = solve_pde(
            ctx=self,
            state=state,
            t_range=t_range,
            dt=dt,
            solver=solver,
            const_cond=const_cond,
            bc=bc,
            bc_ops=bc_ops,
            grid_spec=grid_spec,
        )
        return detailed.solution
