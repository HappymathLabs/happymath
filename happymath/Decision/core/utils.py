"""
决策方法的通用工具函数集合。

本模块提供了所有决策方法类共享的工具函数，
避免代码重复，确保一致性。
"""

import contextlib
import io
import inspect
from typing import Dict, Any, Callable
import matplotlib


@contextlib.contextmanager
def suppress_output():
    """
    上下文管理器：抑制标准输出与错误输出。
    
    使用示例:
        with suppress_output():
            noisy_function()  # 该函数的输出将被抑制
    """
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        yield


def filter_algorithm_params(algorithm_func: Callable, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据算法函数的签名过滤参数，只保留算法支持的参数。
    
    参数:
        algorithm_func: 算法函数
        params: 所有可用的参数字典
        
    返回:
        只包含算法支持参数的字典
    """
    sig = inspect.signature(algorithm_func)
    allowed_params = {p for p in sig.parameters}
    return {k: v for k, v in params.items() if k in allowed_params}


@contextlib.contextmanager
def matplotlib_backend_context(backend: str = 'Agg'):
    """
    临时切换matplotlib后端的上下文管理器。
    
    用于需要抑制图形输出的场景。
    
    参数:
        backend: 要切换到的后端 (默认 'Agg' 为非交互式后端)
        
    使用示例:
        with matplotlib_backend_context():
            plotting_function()  # 图形不会显示
    """
    import matplotlib
    current_backend = matplotlib.get_backend()
    try:
        matplotlib.use(backend)
        yield
    finally:
        # 恢复原始后端（如果可能）
        try:
            matplotlib.use(current_backend)
        except:
            # 某些后端切换可能不被支持，忽略错误
            pass


def execute_algorithm_with_suppression(algorithm_func: Callable, 
                                      params: Dict[str, Any],
                                      needs_plot_suppression: bool = False) -> Any:
    """
    执行算法函数，同时抑制输出和可选的图形显示。
    
    参数:
        algorithm_func: 要执行的算法函数
        params: 算法参数
        needs_plot_suppression: 是否需要抑制图形输出
        
    返回:
        算法执行结果
    """
    # 过滤参数
    filtered_params = filter_algorithm_params(algorithm_func, params)
    
    # 根据需要抑制图形输出
    if needs_plot_suppression:
        with matplotlib_backend_context(), suppress_output():
            result = algorithm_func(**filtered_params)
    else:
        with suppress_output():
            result = algorithm_func(**filtered_params)
    
    return result


def prepare_standard_algorithm_params(base_params: Dict[str, Any] = None,
                                     graph: bool = False,
                                     verbose: bool = False,
                                     **kwargs) -> Dict[str, Any]:
    """
    准备标准算法参数字典。
    
    参数:
        base_params: 基础参数字典
        graph: 是否生成图形 (默认 False)
        verbose: 是否输出详细信息 (默认 False)
        **kwargs: 其他要添加的参数
        
    返回:
        准备好的参数字典
    """
    params = {
        'graph': graph,
        'verbose': verbose
    }
    
    if base_params:
        params.update(base_params)
    
    params.update(kwargs)
    
    return params