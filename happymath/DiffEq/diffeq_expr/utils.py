"""
工具函数模块
从de_base.py迁移的通用工具函数
"""

from typing import Union, List, Dict
import sympy


def eqs2exprs(eqs: Union[sympy.Expr, List]) -> List:
    """
    将微分方程组转换为微分表达式组
    迁移自 de_base._eqs2exprs
    
    Args:
        eqs: 方程或方程组
        
    Returns:
        表达式列表
    """
    def _sympify(eq):
        return list(map(sympy.sympify, eq if hasattr(eq, '__iter__') and not isinstance(eq, str) else [eq]))
    
    eq = _sympify(eqs)
    for i, fi in enumerate(eq):
        if isinstance(fi, sympy.Equality):
            eq[i] = fi.lhs - fi.rhs
    
    return eq


def flatten_list(nested_list: List) -> List:
    """
    将嵌套列表展平 - 迭代版本（避免递归栈溢出）
    迁移自 de_base._flatten
    
    Args:
        nested_list: 嵌套列表
        
    Returns:
        展平的列表
    """
    flat_list = []
    stack = [nested_list]
    
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            # 倒序添加到栈中，保持原有顺序
            stack.extend(reversed(current))
        else:
            flat_list.append(current)
    
    return flat_list


def force_convert_symbols(symbols, sympy_obj, value_range):
    """
    强制转换符号为指定取值范围类型
    迁移自 de_base._force_convert 的思想
    
    Args:
        symbols: 符号集合
        sympy_obj: sympy表达式对象
        value_range: 目标取值范围
        
    Returns:
        (转换后的表达式, 转换字典)
    """
    from diffeq_core.utils import forced_trans_type, check_symbol_type
    from sympy.utilities.iterables import iterable
    
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


def is_number(value) -> bool:
    """
    判断是否为数字
    迁移自 de_base._is_number
    
    Args:
        value: 待判断的对象
        
    Returns:
        是否为数字
    """
    if not isinstance(value, str):
        value = str(value)
    try:
        float(value)
        return True
    except ValueError:
        pass
    
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        pass
    
    return False


def split_expression_meta(expr, mode_list=None) -> List:
    """
    根据运算方式将表达式分离成元表达式
    迁移自 de_base._split_expr_meta
    
    Args:
        expr: 表达式
        mode_list: 分离模式列表
        
    Returns:
        元表达式列表
    """
    from sympy.core.mul import Mul
    from sympy.core.add import Add
    from sympy.core.power import Pow
    
    if mode_list is None:
        mode_list = [Mul, Add, Pow]
    
    expr_meta_list = []
    
    def meta_in_expr(expr_item):
        for arg in expr_item.args:
            if any(isinstance(arg, mode) for mode in mode_list):
                meta_in_expr(arg)
            else:
                expr_meta_list.append(arg)
    
    if hasattr(expr, 'args'):
        meta_in_expr(expr)
    else:
        expr_meta_list.append(expr)
    
    return expr_meta_list


def split_func_vars(func: sympy.Expr):
    """
    分离出函数中的函数名与变量符号
    迁移自 de_base._split_func_vars
    
    Args:
        func: 函数表达式
        
    Returns:
        (函数名, 变量符号)
    """
    func_name = func.func
    var_name = func.args
    return func_name, var_name


def split_derivative_components(der_expr: sympy.Expr):
    """
    分离出导数表达式中的函数名与变量符号
    迁移自 de_base._split_der_funcs_vars
    
    Args:
        der_expr: 导数表达式
        
    Returns:
        (导数函数列表, 函数名列表, 变量名列表)
    """
    from sympy.solvers.ode.ode import _extract_funcs
    
    der_func = _extract_funcs([der_expr])
    func_name = []
    var_name = []
    
    for func in der_func:
        func_name.append(func.func)
        var_name.append(func.args)
    
    return der_func, func_name, var_name


def split_substitution_components(subs_expr):
    """
    分离出subs表达式中的函数名与变量符号
    迁移自 de_base._split_subs_funcs_vars
    
    Args:
        subs_expr: Subs表达式
        
    Returns:
        (函数名, 变量名, 条件值)
    """
    func_name = subs_expr.expr
    var_name = subs_expr.variables
    cond_value = subs_expr.point
    
    return func_name, var_name, cond_value


def rename_dict_key(dictionary: dict, old_key, new_key) -> dict:
    """
    在不改变顺序的前提下重命名字典的key
    
    Args:
        dictionary: 字典
        old_key: 旧键
        new_key: 新键
        
    Returns:
        重命名后的字典
    """
    if old_key in dictionary:
        # 保持顺序的字典重命名
        items = list(dictionary.items())
        new_items = []
        for key, value in items:
            if key == old_key:
                new_items.append((new_key, value))
            else:
                new_items.append((key, value))
        return dict(new_items)
    return dictionary


def extract_expression_properties(expr: sympy.Expr) -> Dict[str, any]:
    """
    提取表达式的基本属性
    
    Args:
        expr: 表达式
        
    Returns:
        属性字典
    """
    return {
        'is_polynomial': expr.is_polynomial() if hasattr(expr, 'is_polynomial') else False,
        'is_rational': expr.is_rational,
        'is_real': expr.is_real,
        'is_complex': expr.is_complex,
        'is_finite': expr.is_finite,
        'is_zero': expr.is_zero,
        'is_positive': expr.is_positive,
        'is_negative': expr.is_negative,
        'free_symbols': expr.free_symbols,
        'atoms_count': len(expr.atoms()),
        'complexity': sympy.count_ops(expr)
    }


def compare_expressions(expr1: sympy.Expr, expr2: sympy.Expr) -> Dict[str, bool]:
    """
    比较两个表达式
    
    Args:
        expr1, expr2: 待比较的表达式
        
    Returns:
        比较结果字典
    """
    return {
        'are_equal': expr1.equals(expr2),
        'are_equivalent': sympy.simplify(expr1 - expr2) == 0,
        'same_structure': expr1.has(*expr2.atoms()) and expr2.has(*expr1.atoms()),
        'same_free_symbols': expr1.free_symbols == expr2.free_symbols
    }