"""
Constraint analyzer.

Parses and analyzes optimization constraints:
- Parse types: Eq, Ge, Gt, Le, Lt, Contains
- Distinguish simple bound vs complex constraints
- Detect discrete variable constraints (FiniteSet)
- Classify constraint categories
- Convert to lambda functions
"""

from typing import List, Tuple, Set, Any, Dict, Optional
from collections.abc import Iterable
import sympy
from sympy import Symbol, lambdify, Eq, Ge, Gt, Le, Lt, FiniteSet, Interval, S, Piecewise
from sympy.sets.contains import Contains
from sympy.logic.boolalg import BooleanTrue, BooleanFalse, Boolean

from ...base.analyzer_base import AnalyzerBase
from ....opt_core.opt_exceptions import ConstraintError, InvalidExpressionError
from ....ir import (
    IRConstraint,
    IRConstraintCategory,
    IRConstraintSense,
    IRDiscreteDomain,
)


class ConstraintAnalyzer(AnalyzerBase):
    """Analyzer for constraints."""

    def __init__(self, constraints):
        """
        Initialize the constraint analyzer.

        Args:
            constraints: Single constraint or list of constraints.
        """
        if constraints is None:
            constraints = []
        elif not isinstance(constraints, Iterable):
            constraints = [constraints]

        super().__init__(constraints)

        self.constraints = list(constraints)
        self._parsed_con_list = []  # parsed constraints list
        self._constraint_counter = 0

        # Validate
        self._validate_constraints()
        self._validate_finite_set_numeric()

        # Parse
        self._parse_constraints()

    def _validate_constraints(self):
        """Check that constraints are of supported types."""
        for i, con in enumerate(self.constraints):
            # Detect BooleanTrue/BooleanFalse (often from simplified numeric constraints)
            if isinstance(con, (BooleanTrue, BooleanFalse)):
                self._raise_boolean_constraint_error(con, i)

            # Allow differential/integral constraints handled by FUNCTIONAL path

            # Check supported constraint types
            if not isinstance(con, (Eq, Ge, Gt, Le, Lt, Contains, Piecewise)):
                raise InvalidExpressionError(
                    expression=con,
                    message=f"Unsupported constraint type: {type(con)}. "
                           f"Supported: Eq, Ge, Gt, Le, Lt, Contains, Piecewise"
                )

    def _contains_differential(self, constraint) -> bool:
        """Check whether constraint contains derivatives."""
        return constraint.has(sympy.Derivative)

    def _contains_integral(self, constraint) -> bool:
        """Check whether constraint contains integrals."""
        return constraint.has(sympy.Integral)

    def _raise_boolean_constraint_error(self, con, index):
        """
        Raise detailed error with hints when constraint simplifies to a boolean.

        Args:
            con: Boolean constraint
            index: Constraint index
        """
        is_true = isinstance(con, BooleanTrue)

        # Compose detailed error message in English
        error_msg = f"""
Constraint[{index}] simplifies to a boolean ({type_name}) due to numeric values.

Explanation:
    When constraints contain concrete numbers, SymPy may evaluate and simplify them.
    Example: If Q[i] = 190.76 (numeric), then Q[i] <= B[i] simplifies to {type_name}.

    Original: {con}
    Simplified: {is_true}

Common reasons:
    1) A variable inside the constraint is actually numeric, not a symbolic variable;
    2) SymPy auto-evaluated the expression;
    3) Mixing numeric lists and symbolic variables in constraints.

Suggestions:
    1) Convert numeric comparisons into bound constraints:
       Not recommended: Q[i] <= B[i]  (if Q[i] is numeric)
       Recommended:     B[i] >= 190.76  (use numeric as a bound)

    2) Use symbolic variables instead of numbers where appropriate.

    3) Inspect constraint construction:
       - Ensure variables are created by symbols();
        - Avoid directly using numeric arrays in relational constraints.

Docs: https://docs.sympy.org/latest/modules/core.html#module-sympy.core.relational
""".format(
            index=index,
            type_name=type(con).__name__,
            con=con,
            is_true=is_true
        )

        raise InvalidExpressionError(
            expression=con,
            message=error_msg
        )

    def _validate_finite_set_numeric(self):
        """Ensure values in FiniteSet constraints can be converted to numbers."""
        for con in self.constraints:
            if isinstance(con, Contains):
                element = con.args[0]
                set_obj = con.args[1]

                if isinstance(set_obj, FiniteSet):
                    # Check each value convertible to numeric
                    non_numeric_values = []
                    for value in set_obj.args:
                        try:
                            float(value)
                        except:
                            non_numeric_values.append(value)

                    if non_numeric_values:
                        raise ConstraintError(
                            constraint=con,
                            message=(
                                f"Variable '{element}' in FiniteSet contains non-numeric values: {non_numeric_values}. "
                                f"All values must be convertible to numeric (int/float). "
                                f"Current values: {list(set_obj.args)}"
                            )
                        )

    def _parse_constraints(self):
        """Parse all constraints (algebraic/domain/logical)."""
        for con in self.constraints:
            parsed = self._parse_single_constraint(con)
            self._parsed_con_list.extend(parsed)

    # Differential/integral constraints handled elsewhere (removed here)

    def _next_identifier(self) -> str:
        """Generate unique constraint identifier."""
        identifier = f"con_{self._constraint_counter}"
        self._constraint_counter += 1
        return identifier

    def _build_relational_constraint(self, constraint) -> IRConstraint:
        """Transform relational constraint into an IR object."""
        con_type = type(constraint)
        free_symbols = tuple(sorted(list(constraint.free_symbols), key=lambda s: str(s)))
        normalized_expr = (constraint.lhs - constraint.rhs).expand()

        if free_symbols:
            lambda_func = lambdify(free_symbols, normalized_expr, "numpy")
        else:
            # For constant constraints, still provide a lambda for uniform handling
            value = float(normalized_expr.evalf())

            def lambda_func(*_args):
                return value

        sense_map = {
            Eq: IRConstraintSense.EQ,
            Ge: IRConstraintSense.GE,
            Gt: IRConstraintSense.GE,
            Le: IRConstraintSense.LE,
            Lt: IRConstraintSense.LE,
        }
        strict = isinstance(constraint, (Gt, Lt))
        metadata = {
            "sympy_type": con_type.__name__,
        }
        if strict:
            metadata["strict_direction"] = "gt" if isinstance(constraint, Gt) else "lt"

        return IRConstraint(
            identifier=self._next_identifier(),
            category=IRConstraintCategory.ALGEBRAIC,
            sense=sense_map.get(con_type),
            lhs=constraint.lhs,
            rhs=constraint.rhs,
            normalized_expr=normalized_expr,
            lambda_func=lambda_func,
            free_symbols=free_symbols,
            strict=strict,
            original=constraint,
            metadata=metadata,
        )

    def _build_discrete_constraint(self, symbol: Symbol, values: Tuple[Any, ...], original) -> IRConstraint:
        """构建离散域约束的IR对象"""
        domain = IRDiscreteDomain(values=values)
        return IRConstraint(
            identifier=self._next_identifier(),
            category=IRConstraintCategory.DOMAIN,
            free_symbols=(symbol,),
            discrete_domain=domain,
            original=original,
            metadata={
                "domain_type": "FiniteSet",
            },
        )

    # 功能型约束构建已移除（当前版本不支持功能型约束）

    def _parse_single_constraint(self, constraint) -> List[IRConstraint]:
        """
        解析单个约束条件

        Returns:
            List[IRConstraint]: 解析后的约束对象列表
        """
        parsed_con_list = []

        # 处理 Piecewise：不再转换为 Big-M，直接封装为 LOGICAL 约束供后端处理
        if isinstance(constraint, Piecewise):
            ir = self._build_logical_piecewise_constraint(constraint)
            parsed_con_list.append(ir)
            return parsed_con_list

        if isinstance(constraint, (Eq, Ge, Gt, Le, Lt)):
            ir_constraint = self._build_relational_constraint(constraint)
            parsed_con_list.append(ir_constraint)
            return parsed_con_list

        elif isinstance(constraint, Contains):
            element = constraint.args[0]
            set_obj = constraint.args[1]

            if isinstance(set_obj, Interval):
                interval_conditions = []
                # 处理区间的左边界
                if set_obj.start is not S.NegativeInfinity:
                    if set_obj.left_open:
                        interval_conditions.append(Gt(element, set_obj.start))
                    else:
                        interval_conditions.append(Ge(element, set_obj.start))

                # 处理区间的右边界
                if set_obj.end is not S.Infinity:
                    if set_obj.right_open:
                        interval_conditions.append(Lt(element, set_obj.end))
                    else:
                        interval_conditions.append(Le(element, set_obj.end))

                for cond in interval_conditions:
                    # 递归解析从Interval产生的不等式
                    parsed_con_list.extend(self._parse_single_constraint(cond))
                return parsed_con_list

            elif isinstance(set_obj, FiniteSet):
                # 约束形式: element ∈ {val1, val2, ...}
                # 当前假设'element'是一个单一的Sympy Symbol
                if not isinstance(element, Symbol):
                    raise ConstraintError(
                        constraint=constraint,
                        message=f"FiniteSet约束中的element必须是单个Sympy Symbol，得到 {type(element)}: {element}. "
                               f"如果element是表达式，需要高级处理（如析取规划）"
                    )

                ir_constraint = self._build_discrete_constraint(
                    symbol=element,
                    values=tuple(set_obj.args),
                    original=constraint
                )
                parsed_con_list.append(ir_constraint)
                return parsed_con_list

            else:
                raise ConstraintError(
                    constraint=constraint,
                    message=f"Contains中不支持的集合类型: {type(set_obj)}"
                )

        else:
            # 如果执行到这里，说明通过了验证但未处理
            raise ConstraintError(
                constraint=constraint,
                message=f"未处理的约束类型: {type(constraint)}"
            )

    def _build_logical_piecewise_constraint(self, piecewise_expr) -> IRConstraint:
        """将 SymPy Piecewise 封装为 LOGICAL IR 约束，供后端解释。

        仅支持 Piecewise 的各分支 expr 为：Eq/Ge/Le/Gt/Lt/True/False
        condition 为 SymPy 布尔表达式（含 Eq(z,0/1) 或区间谓词等）。
        """
        branches = []
        all_syms = set()
        for expr, cond in piecewise_expr.args:
            if not (cond is True or cond is S.true or isinstance(cond, (Eq, Ge, Gt, Le, Lt)) or hasattr(cond, 'free_symbols')):
                raise ConstraintError(
                    constraint=piecewise_expr,
                    message="Piecewise 分支条件必须是布尔表达式（Eq/Ge/Le/Gt/Lt）或 True。"
                )
            # 允许 True/False/Relational 作为分支表达式
            if not (isinstance(expr, (Eq, Ge, Gt, Le, Lt)) or expr in (S.true, S.false)):
                raise ConstraintError(
                    constraint=piecewise_expr,
                    message="Piecewise 分支表达式必须是约束（Eq/Ge/Le/Gt/Lt）或 True/False。"
                )
            branches.append({'expr': expr, 'cond': cond})
            # 收集符号
            if hasattr(expr, 'free_symbols'):
                all_syms.update(expr.free_symbols)
            if hasattr(cond, 'free_symbols'):
                all_syms.update(cond.free_symbols)

        free_symbols = tuple(sorted(list(all_syms), key=lambda s: str(s)))
        return IRConstraint(
            identifier=self._next_identifier(),
            category=IRConstraintCategory.LOGICAL,
            free_symbols=free_symbols,
            original=piecewise_expr,
            metadata={
                'logical_kind': 'piecewise',
                'branches': branches
            }
        )

    # 旧的 Piecewise→Big-M 和二元指示器转换逻辑已移除

    def analyze(self) -> Dict[str, Any]:
        """
        执行分析

        Returns:
            Dict: 包含分析结果的字典
        """
        if self._analyzed:
            return self._analysis_cache

        # 分类约束
        boundary_constraints = []
        discrete_constraints = []
        inequality_constraints = []
        equality_constraints = []
        functional_constraints = []
        all_symbols = set()

        for con_item in self._parsed_con_list:
            all_symbols.update(con_item.iter_symbols())

            if con_item.category == IRConstraintCategory.DOMAIN:
                discrete_constraints.append(con_item)
            elif con_item.category == IRConstraintCategory.FUNCTIONAL:
                functional_constraints.append(con_item)
            elif con_item.category == IRConstraintCategory.ALGEBRAIC:
                if con_item.sense == IRConstraintSense.EQ:
                    equality_constraints.append(con_item)
                else:
                    inequality_constraints.append(con_item)
            else:
                boundary_constraints.append(con_item)

        # 缓存结果
        self._analysis_cache = {
            'parsed_con_list': self._parsed_con_list,
            'boundary_constraints': boundary_constraints,
            'discrete_constraints': discrete_constraints,
            'inequality_constraints': inequality_constraints,
            'equality_constraints': equality_constraints,
            'functional_constraints': functional_constraints,
            'all_symbols': all_symbols,
            'has_integer_variables': len(discrete_constraints) > 0
        }
        self._analyzed = True

        return self._analysis_cache

    def get_symbols(self) -> Set[Symbol]:
        """获取所有符号变量"""
        if not self._analyzed:
            self.analyze()
        return self._analysis_cache['all_symbols']

    def has_integer_variables(self) -> bool:
        """检查是否存在整数/离散变量"""
        if not self._analyzed:
            self.analyze()
        return self._analysis_cache['has_integer_variables']

    # === 属性访问 ===

    @property
    def parsed_con_list(self) -> List[IRConstraint]:
        """获取解析后的约束IR列表"""
        return self._parsed_con_list

    @property
    def discrete_constraints(self) -> List[IRConstraint]:
        """获取离散变量约束IR列表"""
        if not self._analyzed:
            self.analyze()
        return self._analysis_cache['discrete_constraints']

    @property
    def inequality_constraints(self) -> List[IRConstraint]:
        """获取不等式约束IR列表"""
        if not self._analyzed:
            self.analyze()
        return self._analysis_cache['inequality_constraints']

    @property
    def equality_constraints(self) -> List[IRConstraint]:
        """获取等式约束IR列表"""
        if not self._analyzed:
            self.analyze()
        return self._analysis_cache['equality_constraints']

    @property
    def functional_constraints(self) -> List[IRConstraint]:
        """已不支持功能型（微分/积分）约束，始终返回空列表"""
        return []
