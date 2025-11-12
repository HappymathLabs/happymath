"""
Objective function analyzer.

Parses and analyzes optimization objectives:
- Parse objective dict {"min"/"max": expr}
- Extract objective list and senses
- Analyze expression types (linear/quadratic/nonlinear)
- Extract symbols
- Convert to lambda functions
"""

from typing import Dict, List, Set, Any
from sympy import lambdify, Expr, Symbol, Poly, degree, Integral, Derivative
from sympy import sin, cos, tan, exp, log, sqrt
from sympy.core.function import AppliedUndef
from sympy.core.power import Pow

from ...base.analyzer_base import AnalyzerBase
from ....opt_core.opt_exceptions import InvalidExpressionError


class ObjectiveAnalyzer(AnalyzerBase):
    """Objective analyzer"""

    def __init__(self, obj_func: Dict):
        """
        Initialize objective analyzer

        Args:
            obj_func: Objective dict in the form {"min"/"max": expr}.
                     expr can be a single expression or a list of expressions
            
        """
        super().__init__(obj_func)

        # Validate input format
        if not isinstance(obj_func, dict):
            raise InvalidExpressionError(
                expression=obj_func,
                message="obj_func must be a dict with keys 'min' or 'max' and values as SymPy expression(s)"
            )

        self.obj_func_dict = obj_func
        self._senses = []
        self._obj_func_list = []
        self._symbols_list = []  # 每个目标函数对应的符号变量列表
        self._parsed_funcs = []  # lambda函数列表

        # Parse dict
        self._parse_dict()

        # 验证
        self._validate_expressions()

    def _parse_dict(self):
        """解析目标函数字典，只接受 {"min"/"max": expr} 格式"""
        for key, value in self.obj_func_dict.items():
            # key must be 'min' or 'max'
            if not isinstance(key, str) or key not in ("min", "max"):
                raise InvalidExpressionError(
                    message=f"Objective dict key must be 'min' or 'max', got {key}"
                )
            
            sense = key
            expr = value

            # Support single expression or list
            if isinstance(expr, list):
                for single_expr in expr:
                    self._senses.append(sense)
                    self._obj_func_list.append(single_expr)
            else:
                self._senses.append(sense)
                self._obj_func_list.append(expr)

    def _validate_expressions(self):
        """Validate expressions"""
        for i, func in enumerate(self._obj_func_list):
            # Must be a SymPy expression
            if not isinstance(func, Expr):
                raise InvalidExpressionError(
                    expression=func,
                    message=f"Objective must be a SymPy expression, got type {type(func)}"
                )

            # 允许包含积分/微分的功能型目标（FUNCTIONAL），不再抛错
            # 标记逻辑延后在 ParseResult 中处理

    def analyze(self) -> Dict[str, Any]:
        """
        Run analysis and return a result dictionary.

        Returns:
            Dict: Analysis results.
        """
        if self._analyzed:
            return self._analysis_cache

        # Extract symbols
        all_symbols = set()
        for expr in self._obj_func_list:
            expr_symbols = expr.free_symbols
            self._symbols_list.append(expr_symbols)
            all_symbols.update(expr_symbols)

        # Convert to lambda functions
        self._parsed_funcs = self._parse_to_lambdas()

        # Analyze expression types
        expr_types = [self._analyze_single_expression_type(expr) for expr in self._obj_func_list]

        # Cache results
        self._analysis_cache = {
            'obj_func_list': self._obj_func_list,
            'senses': self._senses,
            'symbols': all_symbols,
            'symbols_list': self._symbols_list,
            'parsed_funcs': self._parsed_funcs,
            'expression_types': expr_types
        }
        self._analyzed = True

        return self._analysis_cache

    def get_symbols(self) -> Set[Symbol]:
        """Get all symbols"""
        if not self._analyzed:
            self.analyze()
        return self._analysis_cache['symbols']

    def _parse_to_lambdas(self) -> List:
        """Convert sympy expressions to a list of lambda functions"""
        parsed_funcs_list = []
        for i, func in enumerate(self._obj_func_list):
            # Expand Sum/Product expressions
            func = func.doit()

            # Sort symbols by full string for stability
            symbols = sorted(list(self._symbols_list[i]), key=lambda s: str(s))

            # Check if contains derivative expression
            if func.has(Derivative):
                # Cannot directly convert derivative expressions to lambda;
                # return a placeholder marking special handling
                def differential_placeholder(*args):
                    raise NotImplementedError(
                        f"Differential expression {func} requires special handling and cannot be directly optimized as objective. "
                        f"Consider moving it to constraints or solving the differential equation first."
                    )
                parsed_funcs_list.append(differential_placeholder)
            else:
                try:
                    parsed_func = lambdify(symbols, func)
                    parsed_funcs_list.append(parsed_func)
                except Exception as e:
                    def conversion_error(*args):
                        raise RuntimeError(f"Failed to convert expression {func} to an executable function: {e}")
                    parsed_funcs_list.append(conversion_error)

        return parsed_funcs_list

    # === Expression type analysis ===

    def analyze_expressions_type(self, expressions: List) -> str:
        """
        Analyze the type of a list of expressions (linear/quadratic/nonlinear).

        Args:
            expressions: list of SymPy expressions

        Returns:
            str: "linear", "quadratic", or "nonlinear"
        """
        if not expressions:
            return 'linear'

        max_complexity = 'linear'

        for expr in expressions:
            expr_type = self._analyze_single_expression_type(expr)
            max_complexity = self._get_higher_complexity_type(max_complexity, expr_type)

            # Early-exit if already nonlinear
            if max_complexity == 'nonlinear':
                break

        return max_complexity

    def _analyze_single_expression_type(self, expr: Expr) -> str:
        """
        Analyze the type of a single expression.

        Args:
            expr: SymPy expression

        Returns:
            str: "linear", "quadratic", or "nonlinear"
        """
        try:
            # Expand expression
            expr = expr.expand()

            # 1. Nonlinear functions
            if self._has_nonlinear_functions(expr):
                return 'nonlinear'

            # 2. Free symbols
            symbols = list(expr.free_symbols)

            if not symbols:
                # Constant expression
                return 'linear'

            # 3. Try to form polynomial and analyze degree
            try:
                # 将表达式视为多项式
                poly = Poly(expr, symbols)
                max_degree = poly.total_degree()

                if max_degree <= 1:
                    return 'linear'
                elif max_degree <= 2:
                    # 进一步检查是否真的包含二次项
                    if self._has_quadratic_terms(expr, symbols):
                        return 'quadratic'
                    else:
                        return 'linear'
                else:
                    return 'nonlinear'

            except Exception:
                # 如果无法构造多项式，使用替代方法分析
                return self._analyze_expression_by_structure(expr, symbols)

        except Exception:
            # 如果分析失败，保守地认为是非线性
            return 'nonlinear'

    def _has_nonlinear_functions(self, expr: Expr) -> bool:
        """检查表达式是否包含非线性函数（如三角函数、指数函数等）"""
        # 直接检查表达式中是否包含非线性函数的实例
        if expr.has(sin) or expr.has(cos) or expr.has(tan):
            return True
        if expr.has(exp) or expr.has(log):
            return True
        if expr.has(sqrt):
            return True

        # 检查未定义的函数（AppliedUndef）
        if expr.atoms(AppliedUndef):
            return True

        # 检查分数指数（如 x^(1/2), x^(1/3) 等）
        for atom in expr.atoms(Pow):
            if len(atom.args) == 2:
                base, exponent = atom.args
                if isinstance(base, Symbol):
                    try:
                        exp_val = float(exponent)
                        # 如果指数不是整数，认为是非线性
                        if exp_val != int(exp_val):
                            return True
                    except:
                        # 如果指数包含变量或无法计算，认为是非线性
                        if exponent.free_symbols:
                            return True

        return False

    def _has_quadratic_terms(self, expr: Expr, symbols: List[Symbol]) -> bool:
        """检查表达式是否真的包含二次项"""
        try:
            poly = Poly(expr, symbols)

            # 检查所有单项式的次数
            for monom, coeff in poly.terms():
                total_degree = sum(monom)
                if total_degree == 2:
                    return True

            return False

        except Exception:
            # 使用结构分析作为备选方案
            return self._check_quadratic_by_structure(expr)

    def _check_quadratic_by_structure(self, expr: Expr) -> bool:
        """通过表达式结构检查是否包含二次项"""
        from sympy.core.mul import Mul

        # 检查是否有 x**2 形式的项
        for atom in expr.atoms(Pow):
            if len(atom.args) == 2 and atom.args[1] == 2:
                # 检查底数是否为符号
                if isinstance(atom.args[0], Symbol):
                    return True

        # 检查是否有 x*y 形式的乘积项
        for atom in expr.atoms(Mul):
            symbol_count = 0
            for arg in atom.args:
                if isinstance(arg, Symbol):
                    symbol_count += 1
            if symbol_count >= 2:
                return True

        return False

    def _analyze_expression_by_structure(self, expr: Expr, symbols: List[Symbol]) -> str:
        """通过表达式结构分析类型（当多项式方法失败时使用）"""
        from sympy.core.mul import Mul

        max_power = 1
        has_cross_product = False

        # 遍历表达式中的所有原子
        for atom in expr.atoms():
            if isinstance(atom, Pow):
                base, exp_val = atom.args
                if isinstance(base, Symbol) and base in symbols:
                    try:
                        power = float(exp_val)
                        max_power = max(max_power, power)
                    except:
                        # 如果指数不是数值，认为是非线性
                        return 'nonlinear'

            elif isinstance(atom, Mul):
                # 检查乘积项中的符号数量
                symbol_count = 0
                for arg in atom.args:
                    if isinstance(arg, Symbol) and arg in symbols:
                        symbol_count += 1
                    elif isinstance(arg, Pow) and isinstance(arg.args[0], Symbol):
                        if arg.args[0] in symbols:
                            try:
                                power = float(arg.args[1])
                                symbol_count += power
                            except:
                                return 'nonlinear'

                if symbol_count > 2:
                    return 'nonlinear'
                elif symbol_count == 2:
                    has_cross_product = True

        # 根据分析结果判断类型
        if max_power > 2:
            return 'nonlinear'
        elif max_power == 2 or has_cross_product:
            return 'quadratic'
        else:
            return 'linear'

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
    def obj_func_list(self) -> List:
        """获取目标函数列表"""
        if not self._analyzed:
            self.analyze()
        return self._analysis_cache['obj_func_list']

    @property
    def senses(self) -> List[str]:
        """获取每个目标函数的优化方向"""
        if not self._analyzed:
            self.analyze()
        return self._analysis_cache['senses']

    @property
    def symbols_list(self) -> List[Set]:
        """获取每个目标函数对应的符号变量列表"""
        if not self._analyzed:
            self.analyze()
        return self._analysis_cache['symbols_list']

    @property
    def parsed_funcs(self) -> List:
        """获取解析后的lambda函数列表"""
        if not self._analyzed:
            self.analyze()
        return self._analysis_cache['parsed_funcs']

    @property
    def processed_math_results(self) -> List:
        """获取处理后的数学表达式结果"""
        return self._processed_math_results

    def get_processor(self, expr):
        """获取指定表达式的处理器结果"""
        for result in self._processed_math_results:
            if result.original_expr == expr:
                return result
        return None

    def has_functional_objectives(self) -> bool:
        """检查是否有函数式目标（需要特殊处理的目标）"""
        return len(self._processed_math_results) > 0
