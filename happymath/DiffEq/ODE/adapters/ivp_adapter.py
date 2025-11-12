"""IVP适配器模块

简化的IVP适配器，集成了条件验证和转换功能。
参考PDE的简洁设计，去除了过度工程化的复杂结构。
"""

from typing import Dict, Any, Optional, List, Tuple, Callable, Union
import numpy as np
import sympy
from sympy import lambdify
from happymath.DiffEq.diffeq_core.de_exceptions import (
    ConditionValidationError, InvalidParameterError, 
    FunctionSeparationError, VariableConsistencyError
)
from happymath.DiffEq.ODE.validators import ParameterValidator


class IVPConditionValidator:
    """IVP条件验证器"""
    
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
    def validate_by_order_and_mode(ctx, mode: str) -> None:
        """根据阶数和模式验证定解条件的有效性"""
        if ctx.order == 1 and ctx.derivative_conds != {}:
            raise ConditionValidationError(
                condition="derivative conditions in first-order ODE",
                validation_type="order_constraint",
                reason="一阶常微分方程不应使用导数项作为定解条件"
            )
        
        if ctx.order == 1 and mode == "BVP":
            raise ConditionValidationError(
                condition="BVP mode for first-order ODE",
                validation_type="mode_constraint", 
                reason="一阶常微分方程不能作为边值问题求解"
            )
        
        if ctx.order > 1 and ctx.derivative_conds == {} and mode == "IVP":
            raise ConditionValidationError(
                condition="missing derivative conditions in higher-order IVP",
                validation_type="condition_completeness",
                reason="高阶常微分方程必须包含导数项作为定解条件"
            )
    
    @staticmethod
    def is_number(value) -> bool:
        """检查值是否为数字"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False


class IVPConditionTransformer:
    """IVP条件转换器"""
    
    def __init__(self):
        self.validator = IVPConditionValidator()
    
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
            if mode == "IVP":
                raise ConditionValidationError("The IVP does not support complex boundary conditions.")
        elif "Derivative" in str(cond_expr):
            raise ConditionValidationError(
                condition=str(cond_expr),
                validation_type="derivative_format",
                reason="请使用 '.diff().subs()' 形式定义包含导数项的条件"
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


class IVPAdapter:
    """IVP适配器，专门处理初值问题的适配"""
    
    def __init__(self):
        """初始化IVP适配器"""
        self.validator = IVPConditionValidator()
        self.transformer = IVPConditionTransformer()
    
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
        
        # 验证条件的有效性
        self.validator.validate_by_order_and_mode(ctx, mode)
        
        # 处理常数
        self.transformer.handle_constants(ctx, const_cond)
        
        return function_conds, value_conds
    
    def build_adapter(self, ctx, cond: Dict, const_cond: Optional[Dict] = None) -> Tuple:
        """
        构建 IVP 的 scipy 适配函数及参数
        
        Args:
            ctx: ODE上下文对象
            cond: 条件字典
            const_cond: 常数条件字典
            
        Returns:
            (scipy_ode_func, S_values, const_values) 元组
        """
        # 获取基本信息
        ode_stand = ctx.stand_ode
        ode_stand_expr = [ctx.stand_ode[i].lhs.expr for i in range(len(ctx.stand_ode))]
        Y = ctx.Y_symbols
        
        # 初始化数据结构
        subs_list = []
        subs_dict = {}
        const_dict = {}
        expr_func_list = []
        
        # 预处理条件
        function_conds, value_conds = self.preprocess_conditions(ctx, "IVP", cond, const_cond)
        subs_trans = ctx.non_derivative_conds.copy()
        
        # 获取Y符号的替代字典
        Y_subs_dict = ctx._stand_ode_der_subs(ode_stand, ctx.Y_symbols)
        unknown_constants = ctx.free_consts
        core_symbol = ctx.core_symbol[0]
        
        # 验证并转换初值条件
        for index, value in enumerate(value_conds):
            if self.validator.is_number(value):
                func_trans = function_conds[index](core_symbol)
                original_key = list(ctx.non_derivative_conds.keys())[index]
                subs_trans = self._rename_key(subs_trans, original_key, func_trans)
            else:
                raise ValueError("The initial value condition must be a number.")
        
        # 构建表达式函数列表
        self._build_expression_functions(
            ctx, ode_stand, Y, subs_trans, unknown_constants, const_cond,
            subs_list, subs_dict, const_dict, expr_func_list, Y_subs_dict, core_symbol
        )
        
        # 计算按"函数+导数阶数"排序后的状态顺序，以匹配用户预期（如 x1, x1', x1'', x2, x2', x2''）
        desired_state_order = self._determine_state_order(ctx, Y_subs_dict, list(subs_dict.keys()))
        
        # 如果无法确定，则退回原顺序（稳健性保护）
        state_order = desired_state_order if desired_state_order else ode_stand_expr
        
        # 依据新顺序重排 subs_dict
        subs_dict = {key: subs_dict[key] for key in state_order if key in subs_dict}
        
        # 同步重排 subs_list 与 expr_func_list，使得返回的导数顺序与 S 顺序一致
        # 首先建立每个方程对应的"状态键"（eq_state_key = eq.lhs.expr）
        eq_state_keys = [expr.lhs.expr for expr in ode_stand]
        reordered_subs_list: List[List] = []
        reordered_expr_funcs: List[Callable] = []
        used_indices = set()
        for key in state_order:
            # 找到与该状态对应的方程索引（eq.lhs == Derivative(key, t)）
            target_idx = None
            for idx, expr in enumerate(ode_stand):
                if idx in used_indices:
                    continue
                lhs = expr.lhs
                try:
                    if isinstance(lhs, sympy.Derivative):
                        base = lhs.expr
                        if base == key:
                            target_idx = idx
                            break
                except Exception:
                    continue
            if target_idx is not None:
                reordered_subs_list.append(subs_list[target_idx])
                reordered_expr_funcs.append(expr_func_list[target_idx])
                used_indices.add(target_idx)
        
        # 对于未被包含的（极少数异常情况），保持原顺序追加，避免丢项
        if len(used_indices) != len(expr_func_list):
            for idx in range(len(expr_func_list)):
                if idx not in used_indices:
                    reordered_subs_list.append(subs_list[idx])
                    reordered_expr_funcs.append(expr_func_list[idx])
        
        subs_list = reordered_subs_list
        expr_func_list = reordered_expr_funcs
        
        # 创建scipy兼容的ODE函数
        scipy_ode_func = self._create_scipy_ode_function(
            subs_list, subs_dict, const_dict, expr_func_list
        )
        
        # 返回结果
        if ctx.has_const:
            return scipy_ode_func, list(subs_dict.values()), list(const_dict.values())
        else:
            return scipy_ode_func, list(subs_dict.values()), []
    
    def _build_expression_functions(self, ctx, ode_stand, Y, subs_trans, unknown_constants, const_cond,
                                   subs_list, subs_dict, const_dict, expr_func_list, Y_subs_dict, core_symbol):
        """构建表达式函数列表"""
        for expr in ode_stand:
            expr_rhs = expr.rhs
            sub_list = [core_symbol]  # 第一个总是核心符号

            # 处理Y符号  
            self._process_y_symbols(expr_rhs, Y, Y_subs_dict, ctx,
                                  sub_list=sub_list, subs_dict=subs_dict)
            
            # 处理函数符号
            self._process_function_symbols(expr_rhs, subs_trans, 
                                         sub_list=sub_list, subs_dict=subs_dict)
            
            # 处理常数符号
            self._process_constant_symbols(expr_rhs, unknown_constants, const_cond,
                                         sub_list=sub_list, const_dict=const_dict)

            # 保存替换列表和创建lambda函数
            subs_list.append(sub_list)
            expr_func_list.append(lambdify(sub_list, expr_rhs))
    
    def _process_y_symbols(self, expr_rhs, Y, Y_subs_dict, ctx, sub_list, subs_dict):
        """处理Y符号"""
        for check_Y in Y:
            if expr_rhs.has(check_Y):
                if ctx and ctx.derivative_conds == {}:
                    raise ValueError("There is no derivative condition, please pass in the 'derivative_conds' parameter for solution.")
                if check_Y not in sub_list:
                    sub_list.append(check_Y)
                    if ctx:  # 只有在ctx存在时才处理derivative_conds
                        subs_dict[check_Y] = ctx.derivative_conds[Y_subs_dict[check_Y]][1]
    
    def _process_function_symbols(self, expr_rhs, subs_trans, sub_list, subs_dict):
        """处理函数符号"""
        for func in subs_trans.keys():
            if expr_rhs.has(func):
                if func not in sub_list:
                    sub_list.append(func)
                    subs_dict[func] = subs_trans[func]
    
    def _process_constant_symbols(self, expr_rhs, unknown_constants, const_cond, sub_list, const_dict):
        """处理常数符号"""
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
    
    def _create_scipy_ode_function(self, subs_list, subs_dict, const_dict, expr_func_list) -> Callable:
        """
        创建scipy兼容的ODE函数
        
        Args:
            subs_list: 替换列表
            subs_dict: 替换字典
            const_dict: 常数字典
            expr_func_list: 表达式函数列表
            
        Returns:
            scipy兼容的ODE函数
        """
        def scipy_ode_func(t, S, *constants):
            # 更新状态变量
            for idx, key in enumerate(subs_dict.keys()):
                subs_dict[key] = S[idx]
            
            # 更新常数
            for idx, key in enumerate(const_dict.keys()):
                const_dict[key] = constants[idx] if idx < len(constants) else const_dict[key]

            # 合并所有替换字典
            subs_all_dict = {**subs_dict, **const_dict}
            
            # 构建替换参数
            S_subs = [
                [subs_all_dict[term] for idx, term in enumerate(subs_list[i]) if idx != 0] 
                for i in range(len(subs_list))
            ]
            
            # 计算导数
            return [expr_func_list[idx](t, *S_subs[idx]) for idx in range(len(expr_func_list))]
        
        return scipy_ode_func
    
    def _rename_key(self, dictionary: Dict, old_key: Any, new_key: Any) -> Dict:
        """重命名字典中的键"""
        if old_key in dictionary:
            dictionary[new_key] = dictionary.pop(old_key)
        return dictionary
    
    def _determine_state_order(self, ctx, Y_subs_dict: Dict, state_candidates: List) -> List:
        """确定状态向量的顺序"""
        try:
            # 1) 核心函数按名称排序
            core_funcs_sorted = sorted(ctx.core_func, key=lambda f: str(f.func))

            # 建立 func -> (order -> Y_sym) 的映射
            y_by_func: Dict[Any, Dict[int, Any]] = {}
            for Y_sym, der_expr in Y_subs_dict.items():
                try:
                    if isinstance(der_expr, sympy.Derivative):
                        base_func = der_expr.args[0]
                        der_order = len(der_expr.variables)
                        if base_func not in y_by_func:
                            y_by_func[base_func] = {}
                        y_by_func[base_func][der_order] = Y_sym
                except Exception:
                    continue

            order_keys: List = []
            # 阶段 1: 先添加所有函数本体（位置变量），匹配scipy的[positions, velocities]惯例
            for f in core_funcs_sorted:
                if f in state_candidates:
                    order_keys.append(f)

            # 阶段 2: 再添加所有一阶导数（速度变量）
            for f in core_funcs_sorted:
                if f in y_by_func and 1 in y_by_func[f]:
                    y1 = y_by_func[f][1]
                    if y1 in state_candidates and y1 not in order_keys:
                        order_keys.append(y1)

            # 阶段 3: 二阶及以上导数的 Y，按 ctx.Y_symbols 顺序
            for Y_sym in ctx.Y_symbols:
                der_expr = Y_subs_dict.get(Y_sym)
                try:
                    if isinstance(der_expr, sympy.Derivative):
                        der_order = len(der_expr.variables)
                        if der_order >= 2 and Y_sym in state_candidates and Y_sym not in order_keys:
                            order_keys.append(Y_sym)
                except Exception:
                    continue

            # 补充遗漏项
            residuals = [k for k in state_candidates if k not in order_keys]
            return order_keys + residuals
        except Exception:
            # 任何异常情况下退回空列表，让调用方采用回退顺序
            return []


# 便捷函数，保持向后兼容
def build_ivp_adapter(ctx, cond: Dict, const_cond: Optional[Dict] = None) -> Tuple:
    """
    构建IVP适配器的便捷函数
    
    Args:
        ctx: ODE上下文对象
        cond: 条件字典
        const_cond: 常数条件字典
        
    Returns:
        (scipy_ode_func, S_values, const_values) 元组
        
    Raises:
        ConditionValidationError: 条件验证失败
    """
    try:
        adapter = IVPAdapter()
        return adapter.build_adapter(ctx, cond, const_cond)
    except Exception as e:
        if isinstance(e, (ConditionValidationError, ValueError)):
            raise
        else:
            raise ConditionValidationError(
                condition="IVP adapter building",
                validation_type="build_process", 
                reason=f"IVP适配器构建失败: {str(e)}"
            ) from e


class IVPAdapterBuilder:
    """IVP适配器构建器，保持向后兼容性"""
    
    def __init__(self):
        """初始化IVP适配器构建器"""
        pass
    
    def build_ivp_adapter(self, ctx, cond: Dict, const_cond: Optional[Dict] = None) -> Tuple:
        """构建完整的IVP适配器（向后兼容）"""
        return build_ivp_adapter(ctx, cond, const_cond)
