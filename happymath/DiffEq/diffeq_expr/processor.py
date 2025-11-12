"""
统一表达式处理器
提供统一的入口点来处理ODE和PDE表达式
"""

from typing import Dict, Union, List, Any
import sympy
from sympy.utilities.iterables import iterable

from .DE.analyzers.ode_analyzer import ODEAnalyzer
from .DE.analyzers.pde_analyzer import PDEAnalyzer
from .DE.symbol_managers.ode_symbol_manager import ODESymbolManager
from .DE.symbol_managers.pde_symbol_manager import PDESymbolManager
from .DE.standardizers.ode_standardizer import ODEStandardizer
from .DE.standardizers.pde_standardizer import PDEStandardizer
from .DE.results.ode_result import ODEResult
from .DE.results.pde_result import PDEResult


class ExpressionProcessor:
    """
    统一表达式处理器
    提供统一的入口来处理不同类型的微分方程表达式
    """
    
    def __init__(self):
        """初始化表达式处理器"""
        self.supported_types: Dict[str, Dict] = {
            'ODE': {
                'analyzer': ODEAnalyzer,
                'symbol_manager': ODESymbolManager,
                'standardizer': ODEStandardizer,
                'result': ODEResult,
            },
            'PDE': {
                'analyzer': PDEAnalyzer,
                'symbol_manager': PDESymbolManager,
                'standardizer': PDEStandardizer,
                'result': PDEResult,
            },
        }
    
    def process(self, sympy_obj: Union[sympy.Expr, List], value_range: str = 'real', **kwargs) -> Union[ODEResult, PDEResult]:
        """
        处理微分方程表达式
        
        Args:
            sympy_obj: 微分方程表达式或表达式列表
            value_range: 变量取值范围
            **kwargs: 其他参数（如spatial_var_order等）
            
        Returns:
            处理结果对象（ODEResult或PDEResult）
        """
        try:
            force_type = kwargs.pop('force_type', None)
            # 自动检测表达式类型
            expr_type = self._detect_expression_type(sympy_obj, force_type=force_type, **kwargs)
            
            # 获取对应的处理器组件
            processors = self.supported_types[expr_type]
            
            # 创建分析器
            analyzer_kwargs = {'value_range': value_range}
            if expr_type == 'PDE':
                analyzer_kwargs.update(kwargs)
            analyzer = processors['analyzer'](sympy_obj, **analyzer_kwargs)
            
            # 验证表达式有效性
            if not analyzer.is_valid_expression():
                raise ValueError(f"无效的{expr_type}表达式")
            
            # 创建符号管理器
            symbol_manager = processors['symbol_manager'](analyzer)
            
            # 生成替代符号
            count = symbol_manager.get_substitution_count()
            if count > 0:
                symbol_manager.generate_substitute_symbols(count)
                symbol_manager.create_symbol_mappings()
            
            # 创建标准化器并执行标准化
            standardizer = processors['standardizer'](analyzer, symbol_manager)
            standardizer.standardize()
            
            # 创建结果对象
            result = processors['result'](analyzer, symbol_manager, standardizer)
            
            return result
            
        except Exception as e:
            raise ValueError(f"表达式处理失败: {e}")
    
    def _detect_expression_type(self, sympy_obj: Union[sympy.Expr, List], force_type: str = None, **kwargs) -> str:
        """
        自动检测表达式类型
        
        Args:
            sympy_obj: 表达式对象
            **kwargs: 其他参数
            
        Returns:
            表达式类型字符串
        """
        if force_type:
            preferred = force_type.upper()
            if preferred not in self.supported_types:
                raise ValueError(f"Unsupported force_type: {force_type}")
            analyzer_cls = self.supported_types[preferred]['analyzer']
            analyzer_kwargs = {}
            if preferred == 'PDE':
                analyzer_kwargs.update(kwargs)
            analyzer = analyzer_cls(sympy_obj, **analyzer_kwargs)
            if analyzer.is_valid_expression():
                return preferred
            raise ValueError(f"表达式不是有效的{preferred}")
        
        # 首先尝试ODE检测
        try:
            ode_analyzer = ODEAnalyzer(sympy_obj)
            if ode_analyzer.is_valid_expression():
                return 'ODE'
        except Exception:
            pass
        
        # 然后尝试PDE检测
        try:
            pde_analyzer = PDEAnalyzer(sympy_obj, **kwargs)
            if pde_analyzer.is_valid_expression():
                return 'PDE'
        except Exception:
            pass
        
        # 如果都无法识别，尝试更详细的错误诊断
        error_details = []
        
        try:
            ode_analyzer = ODEAnalyzer(sympy_obj)
            ode_analyzer.is_valid_expression()
        except Exception as e:
            error_details.append(f"ODE检测失败: {e}")
        
        try:
            pde_analyzer = PDEAnalyzer(sympy_obj, **kwargs)
            pde_analyzer.is_valid_expression()  
        except Exception as e:
            error_details.append(f"PDE检测失败: {e}")
        
        # 抛出更详细的错误信息
        error_msg = 'Unable to determine the type of expression, please check the input differential equation expression.'
        if error_details:
            error_msg += f"\n详细错误: {'; '.join(error_details)}"
        
        raise ValueError(error_msg)
    
    def analyze_only(self, sympy_obj: Union[sympy.Expr, List], value_range: str = 'real', **kwargs) -> Dict[str, Any]:
        """
        仅进行表达式分析，不执行完整处理
        
        Args:
            sympy_obj: 微分方程表达式或表达式列表
            value_range: 变量取值范围
            **kwargs: 其他参数
            
        Returns:
            分析结果字典
        """
        try:
            force_type = kwargs.pop('force_type', None)
            expr_type = self._detect_expression_type(sympy_obj, force_type=force_type, **kwargs)
            
            processors = self.supported_types[expr_type]
            analyzer_kwargs = {'value_range': value_range}
            if expr_type == 'PDE':
                analyzer_kwargs.update(kwargs)
            analyzer = processors['analyzer'](sympy_obj, **analyzer_kwargs)
            
            analysis_result = {
                'expression_type': expr_type,
                'is_valid': analyzer.is_valid_expression(),
                'is_linear': analyzer.is_linear,
                'expression_order': analyzer.expression_order,
                'core_functions': analyzer.core_functions,
                'core_symbols': analyzer.core_symbols,
                'derivative_orders': analyzer.derivative_orders,
                'free_constants': analyzer.free_constants,
                'core_func_symbol_mapping': analyzer.core_func_symbol_mapping
            }
            
            # 添加特定类型的信息
            if expr_type == 'ODE':
                analysis_result['is_system'] = (hasattr(analyzer, 'is_system_ode') and 
                                               analyzer.is_system_ode())
            elif expr_type == 'PDE':
                analysis_result['spatial_variables'] = analyzer.spatial_vars
                analysis_result['time_variable'] = analyzer.time_var
                analysis_result['spatial_dimensions'] = len(analyzer.spatial_vars)
                analysis_result['has_mixed_derivatives'] = analyzer.check_mixed_derivatives()
            
            return analysis_result
            
        except Exception as e:
            return {
                'error': str(e),
                'expression_type': 'unknown',
                'is_valid': False
            }
    
    def validate_expression(self, sympy_obj: Union[sympy.Expr, List], **kwargs) -> Dict[str, Any]:
        """
        验证表达式有效性
        
        Args:
            sympy_obj: 微分方程表达式或表达式列表
            **kwargs: 其他参数
            
        Returns:
            验证结果字典
        """
        validation_result = {
            'is_valid': False,
            'expression_type': 'unknown',
            'errors': [],
            'warnings': []
        }
        
        try:
            # 尝试检测类型
            force_type = kwargs.pop('force_type', None)
            expr_type = self._detect_expression_type(sympy_obj, force_type=force_type, **kwargs)
            validation_result['expression_type'] = expr_type
            
            # 创建相应的分析器进行详细验证
            processors = self.supported_types[expr_type]
            analyzer_kwargs = {}
            if expr_type == 'PDE':
                analyzer_kwargs.update(kwargs)
            analyzer = processors['analyzer'](sympy_obj, **analyzer_kwargs)
            
            validation_result['is_valid'] = analyzer.is_valid_expression()
            
            if not validation_result['is_valid']:
                validation_result['errors'].append(f"表达式不是有效的{expr_type}")
            
            # 类型特定的验证
            if expr_type == 'PDE':
                if not analyzer.validate_spatial_dimensions():
                    validation_result['warnings'].append("空间维度可能超过支持范围")
                
                if analyzer.check_mixed_derivatives():
                    validation_result['warnings'].append("包含混合偏导数，可能不被完全支持")
            
        except Exception as e:
            validation_result['errors'].append(str(e))
        
        return validation_result
    
    def get_expression_info(self, sympy_obj: Union[sympy.Expr, List], **kwargs) -> str:
        """
        获取表达式信息的文本摘要
        
        Args:
            sympy_obj: 微分方程表达式或表达式列表
            **kwargs: 其他参数
            
        Returns:
            信息摘要字符串
        """
        try:
            analysis = self.analyze_only(sympy_obj, **kwargs)
            
            if analysis.get('error'):
                return f"表达式分析失败: {analysis['error']}"
            
            info_lines = [
                f"表达式类型: {analysis['expression_type']}",
                f"有效性: {'是' if analysis['is_valid'] else '否'}",
                f"线性性: {'线性' if analysis['is_linear'] else '非线性'}",
                f"阶数: {analysis['expression_order']}",
                f"核心函数数: {len(analysis['core_functions'])}",
                f"核心符号数: {len(analysis['core_symbols'])}",
                f"导数项数: {len(analysis['derivative_orders'])}",
                f"自由常数数: {len(analysis['free_constants'])}"
            ]
            
            # 添加类型特定信息
            if analysis['expression_type'] == 'ODE':
                info_lines.append(f"方程组: {'是' if analysis.get('is_system', False) else '否'}")
            elif analysis['expression_type'] == 'PDE':
                info_lines.extend([
                    f"空间维度: {analysis.get('spatial_dimensions', 0)}",
                    f"时间变量: {analysis.get('time_variable', 'None')}",
                    f"混合偏导数: {'是' if analysis.get('has_mixed_derivatives', False) else '否'}"
                ])
            
            return "\n".join(info_lines)
            
        except Exception as e:
            return f"获取表达式信息失败: {e}"


# 提供便捷的全局处理函数
_default_processor = ExpressionProcessor()

def process_expression(sympy_obj: Union[sympy.Expr, List], **kwargs) -> Union[ODEResult, PDEResult]:
    """
    便捷的全局处理函数
    
    Args:
        sympy_obj: 微分方程表达式或表达式列表
        **kwargs: 其他参数
        
    Returns:
        处理结果对象
    """
    return _default_processor.process(sympy_obj, **kwargs)

def analyze_expression(sympy_obj: Union[sympy.Expr, List], **kwargs) -> Dict[str, Any]:
    """
    便捷的全局分析函数
    
    Args:
        sympy_obj: 微分方程表达式或表达式列表
        **kwargs: 其他参数
        
    Returns:
        分析结果字典
    """
    return _default_processor.analyze_only(sympy_obj, **kwargs)

def validate_expression(sympy_obj: Union[sympy.Expr, List], **kwargs) -> Dict[str, Any]:
    """
    便捷的全局验证函数
    
    Args:
        sympy_obj: 微分方程表达式或表达式列表
        **kwargs: 其他参数
        
    Returns:
        验证结果字典
    """
    return _default_processor.validate_expression(sympy_obj, **kwargs)
