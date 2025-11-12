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
        Note on current limitations:
          1) Spatial dimensions > 2 are not supported.
          2) Mixed derivatives (e.g., ∂²u/∂x∂t) are not supported.
        """
        
        super(PDEModule, self).__init__(sympy_obj, value_range)
        # Base class already holds _sympy_obj; avoid writing to read-only property
        self.range = self._value_range
        self.spatial_var_order = spatial_var_order

        if not self.is_pde:
            raise TypeError("This is not an PDE expression, or this is not a standard PDE expression.")

        # Cache (ExprParser result)
        self._cached_standard_result: Optional[Any] = None
        self._cache_invalid = True
        
        if not self._check_core_symbol(symbol_str="t"):
            raise InvalidExpressionError(sympy_obj, "The PDE must have time variable 't'.")
        
        # Collect spatial variables per list order (x, y, z, ...)
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
        
        # Reorder spatial variables to match spatial_var_order if possible
        self.spatial_order_var_list = [var for var in self.spatial_var_order if var in self.spatial_var_list]
        if len(self.spatial_order_var_list) == len(self.spatial_var_list):
            self.spatial_var_list = self.spatial_order_var_list
        else:
            warnings.warn(f"The spatial variables in the expression do not match the specified order. Will use the order {self.spatial_var_list} as spatial variables.")

    # Check if a core symbol (e.g., time variable 't') exists
    def _check_core_symbol(self, symbol_str: str = "t") -> bool:
        try:
            for symbol in self.core_symbol:
                if str(symbol) == symbol_str:
                    return True
            return False
        except Exception:
            return False

    # Invalidate cache
    def _invalidate_cache(self) -> None:
        self._cache_invalid = True

    # Provide expr property consistent with ODEModule to trigger cache invalidation
    @property
    def expr(self) -> Union[sympy.Expr, list]:
        return self._sympy_obj

    @expr.setter
    def expr(self, new_expr: Union[sympy.Expr, list]):
        self._sympy_obj = new_expr
        self._invalidate_cache()
                       
    # ExprParser: unified standardization entry
    def _compute_standard_pde(self):
        try:
            result = process_expression(self.expr, spatial_var_order=self.spatial_var_order)
            if getattr(result, '_analyzer_result', None) and getattr(result._analyzer_result, 'expression_type', '') != 'PDE':
                raise InvalidExpressionError(self.expr, "This is not a PDE expression.")

            return result
        except Exception as e:
            raise ExpressionStandardizationError(self.expr, "PDE standardization", str(e))

    # Standardized PDE (keeps legacy interface)
    @property
    def stand_pde(self):
        if self._cache_invalid or self._cached_standard_result is None:
            self._cached_standard_result = self._compute_standard_pde()
            self._cache_invalid = False
        return self._cached_standard_result.standardized_expressions
    
    # Convert standardized PDE to a solvable format (provided by ExprParser)
    @property
    def to_solvable_pde(self):
        if self._cache_invalid or self._cached_standard_result is None:
            _ = self.stand_pde  # trigger cache
        return self._cached_standard_result.get_solvable_format() if hasattr(self._cached_standard_result, 'get_solvable_format') else getattr(self._cached_standard_result, 'solvable_format', {})
    
    def ana_solve(self):
        """Analytical solve for PDEs (not implemented)."""
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
        Solve a PDE numerically (facade interface).

        Args:
            state: Initial field. Accepts py-pde Field/FieldCollection, numpy array (single field),
                   or dict[name, np.ndarray] (multi-field). Arrays will use grid_spec to build grid+field.
            t_range: Time range (e.g., (0, 1) or 1.0).
            dt: Time step size.
            const_cond: Constants/coefficient dict; supports scalars, numpy arrays (auto-converted to Field), or existing Field.
            solver: Solver name forwarded to py-pde (e.g., "explicit").
            bc: Boundary conditions (py-pde BoundariesData such as {"x-": {"value": 0}, ...} or "periodic").
            bc_ops: Operator-specific boundary conditions (py-pde bc_ops) used in expressions.
            grid_spec: When state is numpy, grid specification dict with keys like
                       {"bounds": ((xa, xb), ...), "shape": (Nx, ...), "periodic": False}.

        Returns:
            py-pde Trajectory solution object.
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
