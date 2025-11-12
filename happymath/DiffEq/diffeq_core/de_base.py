"""
Refactored DE base class focused on core functionality.
Delegates complex expression analysis and symbol management to dedicated components.
"""

from math import e
from typing import Union, List, Literal, Dict, Any
import sympy
import re
from sympy import classify_ode, classify_pde, dsolve, solve, lambdify, collect, diff, sympify, Basic
from sympy.solvers.ode.ode import solve_ics, _extract_funcs, constant_renumber, classify_sysode
from sympy.solvers.deutils import _preprocess, ode_order, _desolve
from sympy.solvers.ode.systems import _preprocess_eqs
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.add import Add
from sympy.core.power import Pow
from sympy.utilities.iterables import iterable
import logging

from .utils import check_symbol_type, sympy_assumptions, forced_trans_type
from ..diffeq_expr import analyze_expression
from .de_exceptions import (
    InvalidExpressionError
)

class DEBase(Expr):
    """
    Refactored differential equation base class.
    Delegates complex features to dedicated components, focusing on essentials.
    """
    
    def __new__(cls, sympy_obj: Union[Expr, iterable], value_range: str = "real", **kwargs):
        """
        Subclass of sympy.core.expr.Expr with forced type conversion for symbols.

        Args:
            sympy_obj: Differential equation expression or list of expressions.
            value_range: Symbol assumption for variables.
            **kwargs: Extra parameters.
        """
        # Validate value_range
        if value_range not in sympy_assumptions:
            raise ValueError(f"{value_range} is not a valid value range")

        # Normalize input expressions
        if iterable(sympy_obj) and len(sympy_obj) == 1:
            sympy_obj = [item for item in sympy_obj][0]

        # Collect free symbols from expression(s)
        if iterable(sympy_obj):
            symbols = set.union(*(eq.free_symbols for eq in sympy_obj))
        else:
            symbols = sympy_obj.free_symbols

        # Check symbol assumptions
        sign_convert = False
        for symbol in symbols:
            if not check_symbol_type(symbol) == value_range:
                logging.warning(f"{symbol} is not a {value_range} symbol; forced conversion will be applied")
                sign_convert = True

        # Perform forced conversion when needed
        if sign_convert:
            converted_expr, convert_dict = cls._force_convert(symbols, sympy_obj, value_range)
            instance = super().__new__(cls, converted_expr)
            instance._sympy_obj = converted_expr
            instance._converted = True
            instance.convert_dict = convert_dict
            return instance
        else:
            instance = super().__new__(cls, sympy_obj)
            instance._sympy_obj = sympy_obj
            instance._converted = False
            instance.convert_dict = {}
            return instance

    @staticmethod
    def _force_convert(symbols, sympy_obj, value_range):
        """Force-convert symbols in the expression to the target value range type."""
        convert_dict = {}
        if iterable(sympy_obj):
            check_list = []
            for obj in sympy_obj:
                for symbol in symbols:
                    if not check_symbol_type(symbol) == value_range:
                        symbol_subs = forced_trans_type(symbol, value_range)
                        obj = obj.subs({symbol: symbol_subs})
                        convert_dict[symbol] = symbol_subs
                check_list.append(obj)
            return check_list, convert_dict
        else:
            for symbol in symbols:
                if not check_symbol_type(symbol) == value_range:
                    symbol_subs = forced_trans_type(symbol, value_range)
                    sympy_obj = sympy_obj.subs({symbol: symbol_subs})
                    convert_dict[symbol] = symbol_subs
            return sympy_obj, convert_dict

    def __init__(self, sympy_obj: Union[sympy.Expr, iterable], value_range: str = "real"):
        """
        Initialize DEBase.

        Args:
            sympy_obj: Differential equation expression or list.
            value_range: Symbol assumption for variables.
        """
        # Validate value range
        if value_range not in sympy_assumptions:
            raise ValueError(f"{value_range} is not a valid value range")
        else:
            self._value_range = value_range
        
        # Set logger
        self.logger = logging.getLogger(__name__)

    # Override sympy.Expr.free_symbols to support systems
    @property
    def free_symbols(self) -> set:
        """
        Return all free symbols in the expression.
        
        Returns:
            set: Set of free symbols
        """
        if iterable(self._sympy_obj):
            return set.union(*(eq.free_symbols for eq in self._sympy_obj))
        else:
            return self._sympy_obj.free_symbols

    @property
    def expr(self) -> Union[Expr, List[Expr]]:
        """Return the underlying expression object."""
        return self._sympy_obj

    @property
    def is_ode(self) -> bool:
        """
        Determine whether the current expression is an ODE via analyzer.
        
        Returns:
            bool
        """
        try:
            analysis = analyze_expression(self._sympy_obj)
            et = analysis.get('expression_type', 'unknown')
            return et == 'ODE'
        except Exception:
            return False

    @property
    def is_pde(self) -> bool:
        """
        Determine whether the current expression is a PDE via analyzer.
        
        Returns:
            bool
        """
        try:
            analysis = analyze_expression(self._sympy_obj)
            et = analysis.get('expression_type', 'unknown')
            return et == 'PDE'
        except Exception:
            return False

    @property
    def is_linear(self) -> bool:
        """
        Determine whether the differential equation is linear via analyzer.
        
        Returns:
            bool
        """
        try:
            analysis = analyze_expression(self._sympy_obj)
            return analysis.get('is_linear', False)
        except Exception as e:
            self.logger.warning(f"Linear check failed: {e}")
            return False

    @property
    def free_funcs(self) -> set:
        """
        Return set of function variables (delegated to symbol manager).
        
        Returns:
            set
        """
        try:
            analysis = analyze_expression(self._sympy_obj)
            return set(analysis.get('core_functions', []))
        except Exception:
            return set()

    @property
    def free_consts(self) -> set:
        """
        Return set of free constants (delegated to symbol manager).
        
        Returns:
            set
        """
        try:
            analysis = analyze_expression(self._sympy_obj)
            return set(analysis.get('free_constants', []))
        except Exception:
            return set()

    @property
    def core_func(self) -> list:
        """
        Return list of core functions (delegated to symbol manager).
        """
        try:
            analysis = analyze_expression(self._sympy_obj)
            return analysis.get('core_functions', [])
        except Exception:
            return []

    @property
    def core_symbol(self) -> list:
        """
        Return list of core symbols (delegated to symbol manager).
        """
        try:
            analysis = analyze_expression(self._sympy_obj)
            return analysis.get('core_symbols', [])
        except Exception:
            return []

    @property
    def core_func_symbol(self) -> dict:
        """
        Return mapping of core functions to symbols (delegated to symbol manager).
        """
        try:
            analysis = analyze_expression(self._sympy_obj)
            return analysis.get('core_func_symbol_mapping', {})
        except Exception:
            return {}
    
    @property
    def de_order(self) -> dict:
        """
        Return mapping from derivative terms to their orders (delegated).
        """
        try:
            analysis = analyze_expression(self._sympy_obj)
            return analysis.get('derivative_orders', {})
        except Exception:
            return {}

    @property
    def order(self) -> int:
        """
        Return maximum order of the differential equation (delegated).
        
        Returns:
            int
        """
        try:
            analysis = analyze_expression(self._sympy_obj)
            order_val = analysis.get('expression_order', None)
            if order_val is None:
                raise ValueError("Missing order information")
            return int(order_val)
        except Exception as e:
            self.logger.error(f"Failed to get order: {e}")
            raise InvalidExpressionError(self._sympy_obj, f"Unable to determine order: {e}")

    def _eqs2exprs(self, eqs: Union[Expr, iterable]) -> list:
        """
        Convert a system of differential equations to expressions (delegated).
        
        Args:
            eqs: Equation or system of equations
        
        Returns:
            list of expressions
        """
        try:
            from diffeq_expr.utils import eqs2exprs
            return eqs2exprs(eqs)
        except Exception:
            # Fallback: minimal implementation
            try:
                if isinstance(eqs, list):
                    return [sympify(fi.lhs - fi.rhs) if isinstance(fi, sympy.Equality) else sympify(fi) for fi in eqs]
                else:
                    return [sympify(eqs.lhs - eqs.rhs) if isinstance(eqs, sympy.Equality) else sympify(eqs)]
            except Exception:
                return [eqs]

    def _is_number(self, s: Any) -> bool:
        """
        Determine whether an object is numeric.
        
        Args:
            s: Object to test
        
        Returns:
            bool
        """
        if not isinstance(s, str):
            s = str(s)
        try:
            float(s)
            return True
        except ValueError:
            pass
        
        try:
            int(s)
            return True
        except (TypeError, ValueError):
            pass
        
        return False

    def _rename_key(self, dictionary: dict, old_key: Any, new_key: Any) -> dict:
        """
        Rename a key in a dict without changing order.
        
        Args:
            dictionary: The dictionary
            old_key: Old key
            new_key: New key
        
        Returns:
            dict: Updated dict
        """
        if old_key in dictionary:
            # Order-preserving dict rename
            items = list(dictionary.items())
            new_items = []
            for key, value in items:
                if key == old_key:
                    new_items.append((new_key, value))
                else:
                    new_items.append((key, value))
            return dict(new_items)
        return dictionary

    # Keep simplified versions for backward-compatibility
    def _split_func_vars(self, func: Expr):
        """Split function into function name and variable symbols."""
        func_name = func.func
        var_name = func.args
        return func_name, var_name

    def _split_der_funcs_vars(self, der_expr: Expr):
        """Split derivative expression into function name and variable symbols."""
        der_func = _extract_funcs([der_expr])
        func_name = []
        var_name = []
        
        for func in der_func:
            func_name.append(func.func)
            var_name.append(func.args)
        
        return der_func, func_name, var_name

    def _split_subs_funcs_vars(self, subs_expr):
        """Split subs expression into function name, variable and point."""
        func_name = subs_expr.expr
        var_name = subs_expr.variables
        cond_value = subs_expr.point
        
        return func_name, var_name, cond_value

    def _split_expr_meta(self, expr, mode_list=[Mul, Add, Pow]):
        """
        Split expression into meta components per operators (delegated; keep API).
        
        Args:
            expr: Expression
            mode_list: Operator types
        
        Returns:
            list of meta components
        """
        try:
            from ..diffeq_expr.utils import split_expression_meta
            return split_expression_meta(expr, mode_list)
        except Exception as e:
            self.logger.warning(f"Failed to split expression: {e}")
            return [expr]

    # Validation methods
    def validate_expression(self) -> bool:
        """
        Validate expression correctness.
        
        Returns:
            bool
        """
        try:
            # Use unified ExprParser validation for ODE/PDE
            from ..diffeq_expr import validate_expression as _validate_expression
            result = _validate_expression(self._sympy_obj)
            return bool(result.get('is_valid', False))
        except Exception as e:
            self.logger.error(f"Expression validation failed: {e}")
            return False

    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get expression analysis summary.
        
        Returns:
            dict: Summary of analysis
        """
        try:
            analysis = self._expression_analyzer.analyze_expression(self._sympy_obj)
            return {
                'type': analysis.get('type', 'unknown'),
                'is_ode': analysis.get('is_ode', False),
                'is_linear': analysis.get('is_linear', False),
                'order': analysis.get('order', 0),
                'num_functions': len(analysis.get('functions', [])),
                'num_variables': len(analysis.get('variables', [])),
                'has_derivatives': len(analysis.get('derivatives', [])) > 0
            }
        except Exception as e:
            self.logger.error(f"Failed to get analysis summary: {e}")
            return {'error': str(e)}
