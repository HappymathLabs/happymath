"""
约束分析器

负责解析和分析优化问题的约束条件，包括：
- 解析各类约束（Eq, Ge, Gt, Le, Lt, Contains）
- 识别简单边界约束 vs 复杂约束
- 识别离散变量约束（FiniteSet）
- 分类约束类型
- 转换为lambda函数
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
    """约束条件分析器"""

    def __init__(self, constraints):
        """
        初始化约束分析器

        Args:
            constraints: 约束条件，可以是单个约束或约束列表
            
        """
        if constraints is None:
            constraints = []
        elif not isinstance(constraints, Iterable):
            constraints = [constraints]

        super().__init__(constraints)

        self.constraints = list(constraints)
        self._parsed_con_list = []  # 解析后的约束列表
        self._constraint_counter = 0

        # 验证
        self._validate_constraints()
        self._validate_finite_set_numeric()

        # 解析
        self._parse_constraints()

    def _validate_constraints(self):
        """检查约束条件是否是支持的类型"""
        for i, con in enumerate(self.constraints):
            # 检测BooleanTrue/BooleanFalse（通常由数值约束简化而来）
            if isinstance(con, (BooleanTrue, BooleanFalse)):
                self._raise_boolean_constraint_error(con, i)

            # 允许包含微分/积分的功能型约束，改由 FUNCTIONAL 路径处理

            # 检查是否为支持的约束类型
            if not isinstance(con, (Eq, Ge, Gt, Le, Lt, Contains, Piecewise)):
                raise InvalidExpressionError(
                    expression=con,
                    message=f"不支持的约束类型: {type(con)}. "
                           f"支持的类型: Eq, Ge, Gt, Le, Lt, Contains, Piecewise"
                )

    def _contains_differential(self, constraint) -> bool:
        """检查约束是否包含微分表达式"""
        return constraint.has(sympy.Derivative)

    def _contains_integral(self, constraint) -> bool:
        """检查约束是否包含积分表达式"""
        return constraint.has(sympy.Integral)

    def _raise_boolean_constraint_error(self, con, index):
        """
        为布尔约束错误提供详细的错误信息和修复建议

        Args:
            con: 布尔约束
            index: 约束索引
        """
        is_true = isinstance(con, BooleanTrue)

        # 构建详细的错误信息
        error_msg = """
约束[{index}]包含数值，导致约束简化为布尔值 ({type_name})

问题说明:
    当约束表达式中包含具体数值时，SymPy会自动简化约束。
    例如: 如果 Q[i] = 190.76（数值），则 Q[i] <= B[i] 会被简化为 {type_name}

    原始约束: {con}
    简化结果: {is_true}

原因分析:
    这通常发生在以下情况：
    1. 约束中的某个变量实际上是数值而非符号变量
    2. 约束被SymPy自动求值和简化了
    3. 约束中混合了数值列表和符号变量

修复建议:
    1. 将数值约束转换为变量边界:
       不推荐: Q[i] <= B[i]  # 如果 Q[i] 是数值
       推荐:   B[i] >= 190.76  # 直接使用数值作为边界

    2. 或者使用符号变量代替数值:
       n = 6  # 定义变量数量
       Q_symbols = symbols('Q0:{{}}{{}}{{}}' + str(n))  # 创建符号变量数组
       constraints.append(Q_symbols[i] <= B[i])

    3. 检查约束构建代码:
       - 确保所有变量都是符号变量（使用 symbols() 创建）
       - 避免在约束中直接使用数值变量

示例代码:
    # 方法1: 直接使用数值边界（推荐）
    from sympy import symbols
    B = symbols('B0:6')
    Q_pred = [190.76, 28.62, ...]  # 数值列表

    constraints = []
    for i in range(6):
        constraints.append(B[i] >= Q_pred[i])  # 正确
        # 不要写: constraints.append(Q_pred[i] <= B[i])  # 错误

    # 方法2: 使用符号变量
    Q, B = symbols('Q0:6'), symbols('B0:6')
    constraints = [Q[i] <= B[i] for i in range(6)]

详细文档: https://docs.sympy.org/latest/modules/core.html#module-sympy.core.relational
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
        """检查所有FiniteSet约束中的值是否都可以转换为数值类型"""
        for con in self.constraints:
            if isinstance(con, Contains):
                element = con.args[0]
                set_obj = con.args[1]

                if isinstance(set_obj, FiniteSet):
                    # 检查每个值是否可以转换为数值
                    non_numeric_values = []
                    for value in set_obj.args:
                        try:
                            float(value)
                        except:
                            non_numeric_values.append(value)

                    if non_numeric_values:
                        raise ConstraintError(
                            constraint=con,
                            message=f"FiniteSet约束中的变量 '{element}' 包含非数值值: {non_numeric_values}. "
                                   f"所有值必须可转换为数值类型(int, float). "
                                   f"当前值: {list(set_obj.args)}"
                        )

    def _parse_constraints(self):
        """解析所有约束条件（代数/域/逻辑）"""
        for con in self.constraints:
            parsed = self._parse_single_constraint(con)
            self._parsed_con_list.extend(parsed)

    # 已移除对微分/积分约束的处理逻辑

    def _next_identifier(self) -> str:
        """生成约束唯一标识"""
        identifier = f"con_{self._constraint_counter}"
        self._constraint_counter += 1
        return identifier

    def _build_relational_constraint(self, constraint) -> IRConstraint:
        """将对称关系约束转换为IR对象"""
        con_type = type(constraint)
        free_symbols = tuple(sorted(list(constraint.free_symbols), key=lambda s: str(s)))
        normalized_expr = (constraint.lhs - constraint.rhs).expand()

        if free_symbols:
            lambda_func = lambdify(free_symbols, normalized_expr, "numpy")
        else:
            # 常数约束仍提供lambda，便于统一处理
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
