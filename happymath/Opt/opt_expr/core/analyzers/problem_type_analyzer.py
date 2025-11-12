"""
Problem type analyzer.

Identifies problem types:
- Pyomo types (LP/QP/NP/MILP/MIQP/MINP)
- Pymoo types (single/multi/many)
- Features (constraints/discrete variables, etc.)
"""

from typing import Dict, Any, Optional
import numpy as np
import sympy as sp
from sympy import Eq, Ge, Gt, Le, Lt, Contains

from ...base.analyzer_base import AnalyzerBase


class ProblemTypeAnalyzer:
    """Analyzer for optimization problem types."""

    def __init__(self, obj_analyzer, con_analyzer=None):
        """
        Initialize the problem type analyzer.

        Args:
            obj_analyzer: ObjectiveAnalyzer instance.
            con_analyzer: Optional ConstraintAnalyzer instance.
        """
        self.obj_analyzer = obj_analyzer
        self.con_analyzer = con_analyzer

        self._pyomo_type = None
        self._pymoo_type = None
        self._is_convex_qp = None

    def analyze_pyomo_problem_type(self) -> str:
        """
        Analyze Pyomo problem type.

        Returns:
            str: One of 'LP', 'QP', 'NP', 'MILP', 'MIQP', 'MINP'.
        """
        if self._pyomo_type is not None:
            return self._pyomo_type

        obj_func_list = self.obj_analyzer.obj_func_list

        if len(obj_func_list) > 1:
            raise ValueError("Pyomo supports only single-objective problems")

        # 0. If objectives/constraints include integrals/derivatives, treat as nonlinear
        try:
            from sympy import Integral, Derivative
            has_functional = False
            for expr in obj_func_list:
                if getattr(expr, 'has', lambda *_: False)(Integral) or \
                   getattr(expr, 'has', lambda *_: False)(Derivative):
                    has_functional = True
                    break

            if not has_functional and self.con_analyzer is not None:
                for con in getattr(self.con_analyzer, 'constraints', []) or []:
                    try:
                        if (hasattr(con, 'lhs') and (con.lhs.has(Integral) or con.lhs.has(Derivative))) or \
                           (hasattr(con, 'rhs') and (con.rhs.has(Integral) or con.rhs.has(Derivative))):
                            has_functional = True
                            break
                        if hasattr(con, 'has') and (con.has(Integral) or con.has(Derivative)):
                            has_functional = True
                            break
                    except Exception:
                        continue

            if has_functional:
                overall_type = 'nonlinear'
                has_integer_vars = False
                if self.con_analyzer:
                    has_integer_vars = self.con_analyzer.has_integer_variables()
                self._pyomo_type = 'MINP' if has_integer_vars else 'NP'
                return self._pyomo_type
        except Exception:
            # Conservative: if error, continue normal classification
            pass

        # 1. Check integer/discrete variables
        has_integer_vars = False
        if self.con_analyzer:
            has_integer_vars = self.con_analyzer.has_integer_variables()

        # 2. Analyze objective type
        obj_func_type = self.obj_analyzer.analyze_expressions_type(obj_func_list)

        # 3. Analyze constraint type
        constraint_expressions = []
        if self.con_analyzer is not None:
            for con in self.con_analyzer.constraints:
                if isinstance(con, (Eq, Ge, Gt, Le, Lt)):
                    # For (in)equality, reduce to lhs - rhs
                    constraint_expressions.append(con.lhs - con.rhs)
                elif isinstance(con, Contains):
                    # Contains may include expressions
                    element = con.args[0]
                    if hasattr(element, 'free_symbols') and element.free_symbols:
                        constraint_expressions.append(element)

        constraint_type = 'linear'
        if constraint_expressions:
            constraint_type = self.obj_analyzer.analyze_expressions_type(constraint_expressions)

        # 4. Combine: pick higher complexity between objectives and constraints
        overall_type = self._get_higher_complexity_type(obj_func_type, constraint_type)

        # Convexity check only for candidate continuous QP
        self._is_convex_qp = False
        if overall_type == 'quadratic' and not has_integer_vars and constraint_type == 'linear':
            try:
                target_expr = obj_func_list[0]
            except Exception:
                target_expr = None
            if target_expr is not None:
                self._is_convex_qp = self._is_objective_convex_quadratic(target_expr)

        # 5. Final type by presence of integer variables
        if has_integer_vars:
            if overall_type == 'linear':
                self._pyomo_type = 'MILP'
            elif overall_type == 'quadratic':
                self._pyomo_type = 'MIQP'
            else:  # nonlinear
                self._pyomo_type = 'MINP'
        else:
            if overall_type == 'linear':
                self._pyomo_type = 'LP'
            elif overall_type == 'quadratic':
                self._pyomo_type = 'QP'
            else:  # nonlinear
                self._pyomo_type = 'NP'

        return self._pyomo_type

    def analyze_pymoo_problem_type(self) -> Dict[str, Any]:
        """
        Analyze Pymoo problem type and features.

        Returns:
            Dict with keys:
                - 'objective_type': 'single'|'multi'|'many'
                - 'has_constraints': bool
                - 'n_objectives': int
                - 'n_constraints': int
                - 'has_discrete_vars': bool
        """
        if self._pymoo_type is not None:
            return self._pymoo_type

        obj_func_list = self.obj_analyzer.obj_func_list
        n_objectives = len(obj_func_list)

        # Count constraints
        n_constraints = 0
        if self.con_analyzer:
            n_constraints = len(self.con_analyzer.parsed_con_list)

        has_constraints = n_constraints > 0

        # Discrete variables present
        has_discrete_vars = False
        if self.con_analyzer:
            has_discrete_vars = self.con_analyzer.has_integer_variables()

        # Determine by number of objectives
        if n_objectives == 1:
            objective_type = 'single'
        elif n_objectives <= 3:
            objective_type = 'multi'
        else:
            objective_type = 'many'

        self._pymoo_type = {
            'objective_type': objective_type,
            'has_constraints': has_constraints,
            'n_objectives': n_objectives,
            'n_constraints': n_constraints,
            'has_discrete_vars': has_discrete_vars
        }

        return self._pymoo_type

    def _is_objective_convex_quadratic(self, expr) -> bool:
        """Check if objective is convex quadratic (Hessian PSD)."""
        symbols = sorted(list(expr.free_symbols), key=lambda s: str(s))
        if not symbols:
            return True
        try:
            hessian = sp.hessian(expr, symbols)
            if hessian.rows == 0 or hessian.cols == 0:
                return True
            h_list = hessian.tolist()
            h_numeric = np.array([[float(term) for term in row] for row in h_list], dtype=float)
        except Exception:
            return False
        if h_numeric.size == 0:
            return True
        sym_h = 0.5 * (h_numeric + h_numeric.T)
        try:
            eigenvalues = np.linalg.eigvalsh(sym_h)
        except Exception:
            return False
        return bool(np.all(eigenvalues >= -1e-9))

    @staticmethod
    def _get_higher_complexity_type(type1: str, type2: str) -> str:
        """Return the higher-complexity type among two types."""
        complexity_order = {'linear': 1, 'quadratic': 2, 'nonlinear': 3}

        if complexity_order[type1] >= complexity_order[type2]:
            return type1
        else:
            return type2

    # === 属性访问 ===

    @property
    def pyomo_problem_type(self) -> str:
        """获取Pyomo问题类型"""
        if self._pyomo_type is None:
            return self.analyze_pyomo_problem_type()
        return self._pyomo_type

    @property
    def pymoo_problem_type(self) -> Dict[str, Any]:
        """获取Pymoo问题类型"""
        if self._pymoo_type is None:
            return self.analyze_pymoo_problem_type()
        return self._pymoo_type

    @property
    def is_convex_qp(self) -> bool:
        """返回是否识别为凸QP"""
        if self._is_convex_qp is None:
            self.analyze_pyomo_problem_type()
        return bool(self._is_convex_qp)
