"""BVP适配器模块

简化的BVP适配器，集成了边界处理、条件验证和转换功能。
参考PDE的简洁设计，去除了过度工程化的复杂结构。
"""

from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from abc import ABC, abstractmethod
import numpy as np
import sympy
from sympy import lambdify
from functools import partial
from collections import defaultdict
from happymath.DiffEq.diffeq_core.de_exceptions import (
    ConditionValidationError, InvalidParameterError,
    BoundaryConditionError, VariableConsistencyError
)


class BVPValidationError(Exception):
    """BVP验证错误"""
    pass


class BVPConditionValidator:
    """BVP条件验证器"""
    
    @staticmethod
    def validate_mode(mode: str) -> None:
        """验证求解模式参数"""
        if not (mode == "IVP" or mode == "BVP"):
            raise InvalidParameterError(
                "mode", mode, valid_values=["IVP", "BVP"]
            )
    
    @staticmethod
    def validate_condition_dict(cond: Any) -> None:
        """验证条件字典的基本格式"""
        if not isinstance(cond, dict):
            raise InvalidParameterError(
                "cond", type(cond).__name__, expected_type="dict"
            )
    
    @staticmethod
    def validate_const_condition_dict(const_cond: Optional[Dict]) -> None:
        """验证常数条件字典"""
        if const_cond is not None and not isinstance(const_cond, dict):
            raise InvalidParameterError(
                "const_cond", type(const_cond).__name__, expected_type="dict"
            )
    
    @staticmethod
    def validate_bvp_conditions(ctx):
        """验证BVP条件"""
        # 需要先触发 stand_ode 以填充 ctx.undeter_terms，供 _is_ivp_bvp 判定使用
        _ = ctx.stand_ode
        if ctx._is_ivp_bvp != "BVP":
            raise BVPValidationError("The incoming solution conditions are not boundary conditions, please check and try again.")

        # 判断定解条件是否与第三类边界条件重复
        check_bvp_list = [
            *ctx.non_derivative_conds.keys(), 
            *ctx.org_derivative_conds.keys()
        ]
        
        if any([
            check_bvp_list[j] in ctx._split_expr_meta(ctx.third_conds[i])
            for i in range(len(ctx.third_conds)) 
            for j in range(len(check_bvp_list))
        ]):
            raise BVPValidationError("The solution conditions appear repeatedly in the third category of boundary conditions.")
    
    @staticmethod
    def is_number(value) -> bool:
        """检查值是否为数字"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False


class BVPConditionTransformer:
    """BVP条件转换器"""
    
    def __init__(self):
        self.validator = BVPConditionValidator()
    
    def prepare_condition_buckets(self, ctx):
        """初始化在 ctx 上保存的条件容器"""
        ctx.has_const = None
        ctx.non_derivative_conds = {}
        ctx.derivative_conds = {}
        ctx.org_derivative_conds = {}
        ctx.third_conds = []
    
    def split_conditions(self, ctx, mode: str, cond: Dict) -> Tuple[List, List]:
        """将 cond 分解为三类条件，并写回到 ctx"""
        function_conds = []
        value_conds = []
        core_symbol = ctx.core_symbol[0]
        
        try:
            for cond_expr, cond_value in cond.items():
                self._process_single_condition(
                    ctx, mode, cond_expr, cond_value, cond, 
                    core_symbol, function_conds, value_conds
                )
        except ConditionValidationError:
            raise
        except Exception as e:
            raise ConditionValidationError(f"Condition processing failed: {str(e)}") from e
        
        return function_conds, value_conds
    
    def _process_single_condition(self, ctx, mode: str, cond_expr, cond_value, cond: Dict, 
                                core_symbol, function_conds: List, value_conds: List):
        """处理单个条件"""
        if isinstance(cond_expr, sympy.Subs):
            self._process_substitution_condition(ctx, cond_expr, cond_value, core_symbol)
        elif isinstance(cond_expr, (sympy.Add, sympy.Mul, sympy.Pow)) or \
             isinstance(cond_value, (sympy.Add, sympy.Mul, sympy.Pow)):
            self._process_complex_condition(ctx, mode, cond_expr, cond_value, function_conds, value_conds)
        elif "Derivative" in str(cond_expr):
            raise ConditionValidationError(
                condition=str(cond_expr),
                validation_type="derivative_format",
                reason="Please define derivative conditions using the form '.diff(...).subs(...)'"
            )
        else:
            self._process_simple_condition(ctx, cond_expr, cond_value, function_conds, value_conds)
    
    def _process_substitution_condition(self, ctx, cond_expr: sympy.Subs, cond_value, core_symbol):
        """处理替换条件（导数条件）"""
        func_subs, var_subs, cond_subs = ctx._split_subs_funcs_vars(cond_expr)
        var_subs = var_subs[0]
        cond_subs = cond_subs[0]
        
        if str(var_subs) != str(core_symbol):
            raise VariableConsistencyError(
                expected=str(core_symbol), actual=str(var_subs)
            )
        
        if ctx._converted:
            func_subs = func_subs.subs(ctx.convert_dict)
        
        ctx.derivative_conds[func_subs] = (cond_subs, cond_value)
        ctx.org_derivative_conds[cond_expr] = cond_value
    
    def _process_complex_condition(self, ctx, mode: str, cond_expr, cond_value, 
                                 function_conds: List, value_conds: List):
        """处理复杂条件（包含多项的条件）"""
        meta_list = ctx._split_expr_meta(cond_expr)
        func_type_list = [type(func) for func in ctx.core_func]
        item_type_list = [type(item) for item in meta_list]

        # 第三类边值条件
        if self._is_third_type_boundary_condition(meta_list, func_type_list, item_type_list):
            self._process_third_type_condition(ctx, mode, cond_expr, cond_value)
        # 函数型初值条件 / 第一类边值条件
        elif self._has_function_terms(func_type_list, item_type_list):
            self._process_function_condition(ctx, cond_expr, cond_value, function_conds, value_conds)
        # 导数项初值条件 / 第二类边值条件
        elif self._has_substitution_terms(meta_list):
            self._process_derivative_condition(ctx, cond_expr, cond_value)
        else:
            raise ConditionValidationError("There are no derivative items or function items in the key values in the 'cond' dictionary.")
    
    def _is_third_type_boundary_condition(self, meta_list: List, func_type_list: List, item_type_list: List) -> bool:
        """判断是否为第三类边界条件"""
        has_subs = any(isinstance(item, sympy.Subs) for item in meta_list)
        has_func = any(set(func_type_list) & set(item_type_list))
        return has_subs and has_func
    
    def _has_function_terms(self, func_type_list: List, item_type_list: List) -> bool:
        """判断是否包含函数项"""
        return any(set(func_type_list) & set(item_type_list))
    
    def _has_substitution_terms(self, meta_list: List) -> bool:
        """判断是否包含替换项"""
        return any(isinstance(item, sympy.Subs) for item in meta_list)
    
    def _process_third_type_condition(self, ctx, mode: str, cond_expr, cond_value):
        """处理第三类边界条件"""
        standard_expr = cond_expr - cond_value
        func_expr = [i for i in ctx._split_expr_meta(standard_expr) 
                    if any(type(core_func) == type(i) for core_func in ctx.core_func)]
        subs_expr = [i for i in ctx._split_expr_meta(standard_expr) 
                    if isinstance(i, sympy.Subs)]
        
        third_cond_dict = ctx._select_conds(func_expr, subs_expr, [])
        if len(third_cond_dict.keys()) != 1:
            raise ConditionValidationError(
                condition="third type boundary conditions",
                validation_type="boundary_consistency",
                reason="Inconsistent boundary evaluation points for third-type conditions"
            )
        
        ctx.third_conds.append(standard_expr)
    
    def _process_function_condition(self, ctx, cond_expr, cond_value, 
                                  function_conds: List, value_conds: List):
        """处理函数条件"""
        func_expr = [i for i in ctx._split_expr_meta(cond_expr) 
                    if any(type(core_func) == type(i) for core_func in ctx.core_func)]
        
        if len(func_expr) > 1:
            raise ConditionValidationError(
                condition=f"multiple function terms: {func_expr}",
                validation_type="condition_completeness", 
                reason="Too many function terms found in a single boundary/initial condition"
            )
        
        separate_func = func_expr[0]
        separate_res = sympy.solve(sympy.Eq(cond_expr, cond_value), separate_func)
        
        if len(separate_res) == 0:
            raise ConditionValidationError(f"Cannot separate function from condition: {cond_expr}")
        
        ctx.non_derivative_conds[separate_func] = separate_res[0]
        cond_func, value_var = ctx._split_func_vars(separate_func)
        function_conds.append(cond_func)
        value_conds.append(value_var[0])
    
    def _process_derivative_condition(self, ctx, cond_expr, cond_value):
        """处理导数条件"""
        subs_expr = [i for i in ctx._split_expr_meta(cond_expr) 
                    if isinstance(i, sympy.Subs)]
        
        if len(subs_expr) > 1:
            raise ConditionValidationError(
                condition=f"multiple derivative terms: {subs_expr}",
                validation_type="condition_completeness",
                reason="Too many derivative terms found in a single boundary/initial condition"
            )
        
        separate_der = subs_expr[0]
        func_subs, var_subs, cond_subs = ctx._split_subs_funcs_vars(separate_der)
        var_subs = var_subs[0]
        cond_subs = cond_subs[0]
        
        separate_res = sympy.solve(sympy.Eq(cond_expr, cond_value), separate_der)
        if len(separate_res) == 0:
            raise ConditionValidationError(f"Cannot separate derivative from condition: {cond_expr}")
        
        if ctx._converted:
            func_subs = func_subs.subs(ctx.convert_dict)
            separate_der = separate_der.subs(ctx.convert_dict)
        
        ctx.derivative_conds[func_subs] = (cond_subs, separate_res[0])
        ctx.org_derivative_conds[separate_der] = separate_res[0]
    
    def _process_simple_condition(self, ctx, cond_expr, cond_value, 
                                function_conds: List, value_conds: List):
        """处理简单条件（直接的函数条件）"""
        ctx.non_derivative_conds[cond_expr] = cond_value
        cond_func, value_var = ctx._split_func_vars(cond_expr)
        function_conds.append(cond_func)
        value_conds.append(value_var[0])
    
    def handle_constants(self, ctx, const_cond: Optional[Dict]):
        """处理未知常数变量，写回 ctx.has_const"""
        unknown_constants = ctx.free_consts
        
        if len(unknown_constants) > 0:
            ctx.has_const = True
            if const_cond is None:
                raise ValueError("There is a constant term, please pass in the 'const_cond' parameter for solution.")
            
            self.validator.validate_const_condition_dict(const_cond)
            
            for value in const_cond.values():
                if not self.validator.is_number(value):
                    raise TypeError("The value of constant term must be real numbers.")
        else:
            ctx.has_const = False


class BoundaryConditionHandler(ABC):
    """Base class for boundary condition handlers"""
    
    @abstractmethod
    def process(self, bc_expr, boundary_values, subs_order_list, ctx):
        """Process a boundary condition and return residual expression"""
        pass


class NonDerivativeConditionHandler(BoundaryConditionHandler):
    """Handler for non-derivative boundary conditions"""
    
    def __init__(self, non_derivative_conds: Dict):
        self.conditions = non_derivative_conds
    
    def process(self, bc_expr, boundary_values, subs_order_list, ctx):
        """Process non-derivative boundary condition"""
        if type(bc_expr) in subs_order_list:
            sub_num = subs_order_list.index(type(bc_expr))
            return boundary_values[sub_num] - self.conditions[bc_expr]
        else:
            raise ValueError(f"{bc_expr} not in subs_order_list.")


class DerivativeConditionHandler(BoundaryConditionHandler):
    """Handler for derivative boundary conditions"""
    
    def __init__(self, derivative_conds: Dict, org_derivative_conds: Dict):
        self.derivative_conds = derivative_conds
        self.org_derivative_conds = org_derivative_conds
        
    def process(self, bc_expr, boundary_values, subs_order_list, ctx):
        """Process derivative boundary condition"""
        der_func_expr = bc_expr.expr
        
        # 如果有转换字典，应用转换
        if ctx._converted:
            der_func_expr = der_func_expr.subs(ctx.convert_dict)

        if der_func_expr in subs_order_list:
            sub_num = subs_order_list.index(der_func_expr)
            return boundary_values[sub_num] - self.derivative_conds[der_func_expr][1]
        else:
            raise ValueError(f"{der_func_expr} not in subs_order_list.")


class ThirdTypeConditionHandler(BoundaryConditionHandler):
    """Handler for third-type boundary conditions"""
    
    def __init__(self, third_conds: List, third_func_list: List, third_subs_list: List):
        self.conditions = third_conds
        self.func_list = third_func_list
        self.subs_list = third_subs_list
        
    def process(self, bc_expr, boundary_values, subs_order_list, ctx):
        """Process third-type boundary condition"""
        idx_expr = self.conditions.index(bc_expr)
        func_expr = self.func_list[idx_expr]
        subs_expr = self.subs_list[idx_expr]

        substituted_expr = bc_expr
        
        # 处理函数表达式
        if func_expr:
            if type(func_expr) in subs_order_list:
                sub_num = subs_order_list.index(type(func_expr))
                substituted_expr = substituted_expr.subs(func_expr, boundary_values[sub_num])
            else:
                raise ValueError(f"{func_expr} not in subs_order_list.")

        # 处理替换表达式
        if subs_expr:
            for bvp_subs in subs_expr:
                der_func_expr = bvp_subs.expr
                
                # 应用转换（如果有）
                if ctx._converted:
                    der_func_expr = der_func_expr.subs(ctx.convert_dict)
                    
                if der_func_expr in subs_order_list:
                    sub_num = subs_order_list.index(der_func_expr)
                    substituted_expr = substituted_expr.subs(bvp_subs, der_func_expr)
                    substituted_expr = substituted_expr.subs(der_func_expr, boundary_values[sub_num])
                else:
                    raise ValueError(f"{der_func_expr} not in subs_order_list.")
                
        return substituted_expr


class BoundaryConditionProcessor:
    """Unified boundary condition processor"""
    
    def __init__(self, non_derivative_conds: Dict, derivative_conds: Dict, 
                 org_derivative_conds: Dict, third_conds: List, 
                 third_func_list: List, third_subs_list: List):
        """Initialize boundary condition processor"""
        # 创建各种类型的处理器
        self.handlers = {
            'non_derivative': NonDerivativeConditionHandler(non_derivative_conds),
            'derivative': DerivativeConditionHandler(derivative_conds, org_derivative_conds),
            'third_type': ThirdTypeConditionHandler(third_conds, third_func_list, third_subs_list)
        }
        
        # Map boundary condition expression to handler type for quick lookup
        self.condition_map = {}
        
        # Map non-derivative conditions
        for bc_expr in non_derivative_conds.keys():
            self.condition_map[bc_expr] = 'non_derivative'
        
        # Map derivative conditions
        for bc_expr in org_derivative_conds.keys():
            self.condition_map[bc_expr] = 'derivative'
        
        # Map third-type conditions
        for bc_expr in third_conds:
            self.condition_map[bc_expr] = 'third_type'
    
    def process_conditions(self, bc_expr_list: List, boundary_values, subs_order_list, ctx) -> List:
        """Process a list of boundary conditions into residuals"""
        residuals = []
        
        for bc_expr in bc_expr_list:
            handler = self._get_handler(bc_expr)
            residual = handler.process(bc_expr, boundary_values, subs_order_list, ctx)
            residuals.append(residual)
        
        return residuals
    
    def _get_handler(self, bc_expr) -> BoundaryConditionHandler:
        """Get handler for the given boundary condition expression"""
        if bc_expr not in self.condition_map:
            raise ValueError(f"Unknown boundary condition: {bc_expr}")
        
        handler_type = self.condition_map[bc_expr]
        return self.handlers[handler_type]


class BVPAdapter:
    """BVP adapter that prepares solver/boundary-callable for SciPy"""
    
    def __init__(self):
        """初始化BVP适配器"""
        self.validator = BVPConditionValidator()
        self.transformer = BVPConditionTransformer()
    
    def preprocess_conditions(self, ctx, mode: str, cond: Dict, const_cond: Optional[Dict] = None) -> Tuple[List, List]:
        """完整的条件预处理流程"""
        # 验证基本参数
        self.validator.validate_mode(mode)
        self.validator.validate_condition_dict(cond)
        self.validator.validate_const_condition_dict(const_cond)
        
        # 准备条件容器
        self.transformer.prepare_condition_buckets(ctx)
        
        # 分解条件
        function_conds, value_conds = self.transformer.split_conditions(ctx, mode, cond)
        
        # 验证BVP特定条件
        self.validator.validate_bvp_conditions(ctx)
        
        # 处理常数
        self.transformer.handle_constants(ctx, const_cond)
        
        return function_conds, value_conds
    
    def build_adapter(self, ctx, cond: Dict, const_cond: Optional[Dict] = None) -> Tuple:
        """
        构建完整的BVP适配器
        
        Args:
            ctx: ODE上下文对象
            cond: 条件字典
            const_cond: 常数条件字典
            
        Returns:
            (scipy_ode_func, bc_func, S_values, const_values) 元组
        """
        # 预处理条件
        function_conds, value_conds = self.preprocess_conditions(ctx, "BVP", cond, const_cond)
        
        # 准备上下文
        ode_stand = ctx.stand_ode
        ode_stand_expr = [ctx.stand_ode[i].lhs.expr for i in range(len(ctx.stand_ode))]
        Y = ctx.Y_symbols
        core_symbol = ctx.core_symbol[0]
        
        # 构建求解函数
        solver_function, subs_dict = self._create_solver_function(ctx, const_cond)
        
        # 构建边界条件函数
        boundary_function = self._create_boundary_function(ctx, subs_dict)
        
        # 构建结果
        return self._build_result(ctx, solver_function, boundary_function, subs_dict, const_cond)
    
    def _create_solver_function(self, ctx, const_cond: Optional[Dict] = None) -> Tuple[Callable, Dict]:
        """创建BVP求解器函数"""
        ode_stand = ctx.stand_ode
        Y = ctx.Y_symbols
        core_symbol = ctx.core_symbol[0]
        
        subs_list = []
        subs_dict = {}
        const_dict = {}
        expr_func_list = []
        unknown_constants = ctx.free_consts

        for idx, expr in enumerate(ode_stand):
            expr_rhs = expr.rhs
            sub_list = [core_symbol]

            # 处理Y符号
            for check_Y in Y:
                if expr_rhs.has(check_Y):
                    if check_Y not in sub_list:
                        sub_list.append(check_Y)
                        subs_dict[check_Y] = None

            # 处理函数符号
            for func in ctx.non_derivative_conds.keys():
                func_name, var = ctx._split_func_vars(func)
                func_trans = func_name(core_symbol)
                if expr_rhs.has(func_trans):
                    if func_trans not in sub_list:
                        sub_list.append(func_trans)
                        subs_dict[func_trans] = None

            # 处理常数符号
            for const in unknown_constants:
                if expr_rhs.has(const):
                    if const not in sub_list:
                        sub_list.append(const)
                        # 修复：通过符号名称匹配找到对应的常数值
                        const_value = None
                        if const_cond:
                            # 首先尝试直接匹配
                            if const in const_cond:
                                const_value = const_cond[const]
                            else:
                                # 如果直接匹配失败，尝试通过字符串名称匹配
                                const_str = str(const)
                                for key, value in const_cond.items():
                                    if str(key) == const_str:
                                        const_value = value
                                        break
                        
                        if const_value is not None:
                            const_dict[const] = const_value
                        else:
                            raise ValueError(f"Missing constant condition for: {const}")

            subs_list.append(sub_list)
            expr_func_list.append(lambdify(sub_list, expr_rhs))

        # 验证表达式完整性
        ode_stand_expr = [ctx.stand_ode[i].lhs.expr for i in range(len(ctx.stand_ode))]
        for lhs_expr in ode_stand_expr:
            if lhs_expr not in subs_dict.keys():
                raise ValueError(f"The expression {lhs_expr} is not correctly defined.")

        subs_dict = {key: subs_dict[key] for key in ode_stand_expr}

        def _ode_func(t, y, const: list = []):
            y_list = [y[i] for i in range(len(y))]
            const_list = [const[i] for i in range(len(const))]

            for idx, key in enumerate(subs_dict.keys()):
                subs_dict[key] = y_list[idx]
            for idx, key in enumerate(const_dict.keys()):
                const_dict[key] = const_list[idx]

            subs_all_dict = {**subs_dict, **const_dict}
            S_subs = [
                [subs_all_dict[term] for idx, term in enumerate(subs_list[i]) if idx != 0] 
                for i in range(len(subs_list))
            ]
            return [expr_func_list[idx](t, *S_subs[idx]) for idx in range(len(expr_func_list))]

        scipy_ode_func = partial(_ode_func, const=list(const_dict.values()))
        return scipy_ode_func, subs_dict
    
    def _create_boundary_function(self, ctx, subs_dict: Dict) -> Callable:
        """创建边界条件函数"""
        selected_bvp_dict = ctx._select_conds(
            ctx.non_derivative_conds, 
            ctx.org_derivative_conds, 
            ctx.third_conds
        )
        min_bc = min(selected_bvp_dict.keys())
        max_bc = max(selected_bvp_dict.keys())

        # 准备边界条件处理所需的数据
        third_func_list, third_subs_list = self._prepare_third_conditions(ctx.third_conds, ctx)
        
        Y_subs_dict = ctx._stand_ode_der_subs(ctx.stand_ode, ctx.Y_symbols)
        subs_dict_order = [key.subs(Y_subs_dict) for key in subs_dict.keys()]
        subs_order_list = [
            type(item) if not isinstance(item, sympy.Derivative) else item 
            for item in subs_dict_order
        ]

        # 创建边界条件处理器
        boundary_processor = BoundaryConditionProcessor(
            ctx.non_derivative_conds, 
            ctx.derivative_conds, 
            ctx.org_derivative_conds, 
            ctx.third_conds,
            third_func_list, 
            third_subs_list
        )

        def bc(ya, yb):
            residuals = []
            
            # 处理左边界条件 (min_bc)
            left_conditions = selected_bvp_dict.get(min_bc, [])
            if left_conditions:
                left_residuals = boundary_processor.process_conditions(
                    left_conditions, ya, subs_order_list, ctx
                )
                residuals.extend(left_residuals)
            
            # 处理右边界条件 (max_bc)
            right_conditions = selected_bvp_dict.get(max_bc, [])
            if right_conditions:
                right_residuals = boundary_processor.process_conditions(
                    right_conditions, yb, subs_order_list, ctx
                )
                residuals.extend(right_residuals)
            
            return np.array(residuals)
        
        return bc
    
    def _prepare_third_conditions(self, third_conds: List, ctx) -> Tuple[List, List]:
        """准备第三类边界条件数据"""
        third_func_list = []
        third_subs_list = []
        
        for third_key in third_conds:
            func_expr = [
                i for i in ctx._split_expr_meta(third_key) 
                if any(type(core_func) == type(i) for core_func in ctx.core_func)
            ]
            third_func_list.append(func_expr[0] if func_expr else None)
            
            subs_expr = [
                i for i in ctx._split_expr_meta(third_key) 
                if isinstance(i, sympy.Subs)
            ]
            third_subs_list.append(subs_expr)
        
        return third_func_list, third_subs_list
    
    def _build_result(self, ctx, solver_func: Callable, bc_func: Callable, 
                     subs_dict: Dict, const_cond: Optional[Dict]) -> Tuple:
        """构建最终结果"""
        if ctx.has_const:
            const_dict = {}
            # 修复：使用符号名称匹配来处理常数字典
            for const in ctx.free_consts:
                if const_cond:
                    const_value = None
                    # 首先尝试直接匹配
                    if const in const_cond:
                        const_value = const_cond[const]
                    else:
                        # 如果直接匹配失败，尝试通过字符串名称匹配
                        const_str = str(const)
                        for key, value in const_cond.items():
                            if str(key) == const_str:
                                const_value = value
                                break
                    
                    if const_value is not None:
                        const_dict[const] = const_value
            
            return solver_func, bc_func, list(subs_dict.values()), list(const_dict.values())
        else:
            return solver_func, bc_func, list(subs_dict.values()), []


# 便捷函数，保持向后兼容
def build_bvp_adapter(ctx, cond: Dict, const_cond: Optional[Dict] = None) -> Tuple:
    """
    构建BVP适配器的便捷函数
    
    Args:
        ctx: ODE上下文对象
        cond: 条件字典
        const_cond: 常数条件字典
        
    Returns:
        (scipy_ode_func, bc_func, S_values, const_values) 元组
        
    Raises:
        ConditionValidationError: 条件验证失败
        BVPValidationError: BVP验证失败
    """
    try:
        adapter = BVPAdapter()
        return adapter.build_adapter(ctx, cond, const_cond)
    except (BVPValidationError, ConditionValidationError):
        raise
    except Exception as e:
        raise ConditionValidationError(
            condition="BVP adapter building",
            validation_type="build_process", 
            reason=f"BVP适配器构建失败: {str(e)}"
        ) from e


class BVPAdapterBuilder:
    """BVP适配器构建器，保持向后兼容性"""
    
    def __init__(self):
        """初始化BVP适配器构建器"""
        pass
    
    def build_bvp_adapter(self, ctx, cond: Dict, const_cond: Optional[Dict] = None) -> Tuple:
        """构建完整的BVP适配器（向后兼容）"""
        return build_bvp_adapter(ctx, cond, const_cond)
