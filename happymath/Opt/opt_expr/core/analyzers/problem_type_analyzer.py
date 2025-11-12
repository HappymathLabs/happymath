"""
问题类型分析器

负责识别优化问题的类型，包括：
- Pyomo问题类型（LP/QP/NP/MILP/MIQP/MINP）
- Pymoo问题类型（single/multi/many）
- 问题特征（约束/离散变量等）
"""

from typing import Dict, Any, Optional
import numpy as np
import sympy as sp
from sympy import Eq, Ge, Gt, Le, Lt, Contains

from ...base.analyzer_base import AnalyzerBase


class ProblemTypeAnalyzer:
    """问题类型分析器"""

    def __init__(self, obj_analyzer, con_analyzer=None):
        """
        初始化问题类型分析器

        Args:
            obj_analyzer: ObjectiveAnalyzer实例
            con_analyzer: ConstraintAnalyzer实例（可选）
        """
        self.obj_analyzer = obj_analyzer
        self.con_analyzer = con_analyzer

        self._pyomo_type = None
        self._pymoo_type = None
        self._is_convex_qp = None

    def analyze_pyomo_problem_type(self) -> str:
        """
        分析Pyomo问题类型

        Returns:
            str: 问题类型缩写
                - 'LP': Linear programming
                - 'QP': Quadratic programming
                - 'NP': Nonlinear programming
                - 'MILP': Mixed-integer linear programming
                - 'MIQP': Mixed-integer quadratic programming
                - 'MINP': Mixed-integer nonlinear programming
        """
        if self._pyomo_type is not None:
            return self._pyomo_type

        obj_func_list = self.obj_analyzer.obj_func_list

        if len(obj_func_list) > 1:
            raise ValueError("Pyomo仅支持单目标优化问题")

        # 0. 若目标/约束包含积分或导数，直接按非线性处理（更稳健，避免LP/QP误判）
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
            # 保守：发生任何异常，继续走常规判定
            pass

        # 1. 检查是否存在整数/离散变量
        has_integer_vars = False
        if self.con_analyzer:
            has_integer_vars = self.con_analyzer.has_integer_variables()

        # 2. 分析目标函数的类型
        obj_func_type = self.obj_analyzer.analyze_expressions_type(obj_func_list)

        # 3. 分析约束条件的类型
        constraint_expressions = []
        if self.con_analyzer is not None:
            for con in self.con_analyzer.constraints:
                if isinstance(con, (Eq, Ge, Gt, Le, Lt)):
                    # 对于等式或不等式约束，提取左边减右边的表达式
                    constraint_expressions.append(con.lhs - con.rhs)
                elif isinstance(con, Contains):
                    # Contains约束中可能包含表达式
                    element = con.args[0]
                    if hasattr(element, 'free_symbols') and element.free_symbols:
                        constraint_expressions.append(element)

        constraint_type = 'linear'
        if constraint_expressions:
            constraint_type = self.obj_analyzer.analyze_expressions_type(constraint_expressions)

        # 4. 综合判断问题类型
        # 取目标函数和约束条件中复杂度更高的类型
        overall_type = self._get_higher_complexity_type(obj_func_type, constraint_type)

        # 凸性检测仅在候选二次连续规划时执行
        self._is_convex_qp = False
        if overall_type == 'quadratic' and not has_integer_vars and constraint_type == 'linear':
            try:
                target_expr = obj_func_list[0]
            except Exception:
                target_expr = None
            if target_expr is not None:
                self._is_convex_qp = self._is_objective_convex_quadratic(target_expr)

        # 5. 根据是否有整数变量确定最终类型
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
        分析Pymoo问题类型

        Returns:
            Dict: 包含问题特征的字典
                - 'objective_type': 'single', 'multi', 或 'many'
                - 'has_constraints': True/False
                - 'n_objectives': 目标函数数量
                - 'n_constraints': 约束数量
                - 'has_discrete_vars': 是否有离散变量
        """
        if self._pymoo_type is not None:
            return self._pymoo_type

        obj_func_list = self.obj_analyzer.obj_func_list
        n_objectives = len(obj_func_list)

        # 约束数量
        n_constraints = 0
        if self.con_analyzer:
            n_constraints = len(self.con_analyzer.parsed_con_list)

        has_constraints = n_constraints > 0

        # 是否有离散变量
        has_discrete_vars = False
        if self.con_analyzer:
            has_discrete_vars = self.con_analyzer.has_integer_variables()

        # 根据目标函数数量确定问题类型
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
        """判断目标是否为凸二次函数（Hessian 半正定）。"""
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
        """返回两个类型中复杂度更高的类型"""
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
