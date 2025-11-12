"""
Objective weighting method collection.

Implements data-driven weighting methods (e.g., CRITIC/Entropy/MEREC/PSI) and
exposes a unified interface for input validation, method dispatch, and results
access in multi-criteria scenarios.
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Any, Union

from ..algorithm import critic, entropy, idocriw, merec, psi_m, seca, cilos

from ..core.base import DecisionBase
from ..core.method_registry import MethodRegistry
from ..core.validators import ParameterValidator
from ..core.utils import execute_algorithm_with_suppression, prepare_standard_algorithm_params
from ..results.result_manager import ResultManager


class ObjWeighting(DecisionBase):
    """
    Collection of objective weighting methods.

    Methods determine criterion weights from the original decision matrix by
    analyzing data structure. Provides unified invocation and result comparison/aggregation.
    """
    
    # 客观权重方法的注册表
    _METHOD_REGISTRY = MethodRegistry.OBJ_WEIGHTING_METHODS
    
    # 方法到其必需参数的映射
    _METHOD_MAP = {
        'critic': ['dataset', 'criterion_type'],
        'entropy': ['dataset', 'criterion_type'],
        'idocriw': ['dataset', 'criterion_type'],
        'merec': ['dataset', 'criterion_type'],
        'mpsi': ['dataset', 'criterion_type'],
        'seca': ['dataset', 'criterion_type'],
        'cilos': ['dataset', 'criterion_type']
    }
    
    _ALGORITHM_MAP = {
        'critic': critic,
        'entropy': entropy,
        'idocriw': idocriw,
        'merec': merec,
        'mpsi': psi_m.mpsi,
        'seca': seca,
        'cilos': cilos
    }
    
    def __init__(self, methods: Optional[Union[str, List[str]]] = None):
        """
        Initialize objective weighting method collection.

        Args:
            methods: Method name or list to execute. None to auto-select available methods.
        """
        super().__init__(methods)
        self.result_manager = ResultManager()
    
    def _validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """Validate and normalize inputs for objective weighting methods."""
        validated_params = {}
        
        # 提取关键参数
        dataset = kwargs.get('dataset')
        criterion_type = kwargs.get('criterion_type')
        
        # 校验并标准化决策矩阵
        if dataset is not None:
            validation_result = ParameterValidator.validate_decision_matrix(dataset)
            if not validation_result['is_valid']:
                raise ValueError(validation_result['error_message'])
            validated_params['dataset'] = validation_result['processed_data']
        
        # 校验准则类型
        if criterion_type is not None:
            n_criteria = validated_params['dataset'].shape[1] if 'dataset' in validated_params else None
            criterion_type_result = ParameterValidator.validate_criterion_type(
                criterion_type, n_criteria=n_criteria
            )
            if not criterion_type_result['is_valid']:
                raise ValueError(criterion_type_result['error_message'])
            validated_params['criterion_type'] = criterion_type_result['processed_data']
        
        return validated_params
    
    def _execute_method(self, method_name: str, **kwargs) -> Any:
        """
        Dispatch and execute a specific objective weighting algorithm.

        Args:
            method_name: Method name.
            **kwargs: Validated parameters.

        Returns:
            Execution result.
        """
        if method_name not in self._ALGORITHM_MAP:
            raise ValueError(f"Unknown objective weighting method: {method_name}")

        # 构造最终传递给算法的参数字典
        final_params = prepare_standard_algorithm_params()
        
        # 获取所需参数并添加到最终参数中
        required_params = self._METHOD_MAP.get(method_name, [])
        final_params.update({k: kwargs[k] for k in required_params if k in kwargs})

        # 执行算法
        algorithm_func = self._ALGORITHM_MAP[method_name]
        result = execute_algorithm_with_suppression(algorithm_func, final_params)
        
        return result

    def get_all_results(self) -> Dict[str, Any]:
        """Return a dictionary containing all stored results."""
        return self.result_manager.get_all_results()
    
    def get_weights(self, method: Optional[str] = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Get weight vectors computed by methods.

        Args:
            method: Method name. None to return a mapping for all methods.

        Returns:
            Single weight vector or mapping from method name to weight vector.
        """
        if method:
            result = self.results.get(method)
            if result is None:
                raise ValueError(f"No results found for method: {method}")
            
            # 按返回类型提取权重
            if isinstance(result, tuple):
                return result[0]
            elif isinstance(result, np.ndarray):
                return result
            else:
                return result
        else:
            # 返回所有方法的权重
            return self.result_manager.get_all_weights()
    
    def compare_weights(self) -> Any:
        """Compare weights across methods (returns a DataFrame)."""
        return self.result_manager.compare_weights()
    
