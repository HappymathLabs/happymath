"""
Utility functions shared by decision methods.

Provides helpers to reduce code duplication and keep behavior consistent.
"""

import contextlib
import io
import inspect
from typing import Dict, Any, Callable
import matplotlib


@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr.

    Example:
        with suppress_output():
            noisy_function()
    """
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        yield


def filter_algorithm_params(algorithm_func: Callable, params: Dict[str, Any]) -> Dict[str, Any]:
    """Filter params by the signature of an algorithm function.

    Args:
        algorithm_func: The algorithm function to inspect.
        params: All available parameters.

    Returns:
        Dict with only supported parameters.
    """
    sig = inspect.signature(algorithm_func)
    allowed_params = {p for p in sig.parameters}
    return {k: v for k, v in params.items() if k in allowed_params}


@contextlib.contextmanager
def matplotlib_backend_context(backend: str = 'Agg'):
    """Temporarily switch matplotlib backend.

    Useful when plots should be suppressed.

    Args:
        backend: Backend name (default 'Agg' is non-interactive).

    Example:
        with matplotlib_backend_context():
            plotting_function()
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
    """Execute an algorithm function while suppressing output and optionally plots.

    Args:
        algorithm_func: Function to execute.
        params: Parameters for the function.
        needs_plot_suppression: Whether to suppress plotting output as well.

    Returns:
        The algorithm result.
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
    """Prepare a standard parameter dictionary for algorithms.

    Args:
        base_params: Base dictionary to update.
        graph: Whether to generate plots.
        verbose: Whether to print verbose output.
        **kwargs: Extra key-values to include.

    Returns:
        Prepared parameter dictionary.
    """
    params = {
        'graph': graph,
        'verbose': verbose
    }
    
    if base_params:
        params.update(base_params)
    
    params.update(kwargs)
    
    return params
