"""
Abstract base class for all decision methods.

This module provides the common foundation for decision analysis classes:
- Unified interfaces and lifecycle for running methods
- Parameter validation and auto-selection based on available inputs
- Result storage and retrieval utilities
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import numpy as np
import warnings


class DecisionBase(ABC):
    """
    Base class for all decision methods.

    Provides:
    - Unified interface for running methods
    - Smart method selection based on available inputs
    - Result storage and management
    - Parameter validation framework
    """
    
    # 方法注册表：将方法名称映射到其所需参数
    # 子类应重写此属性
    _METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {}
    
    def __init__(self, methods: Optional[Union[str, List[str]]] = None):
        """
        Initialize the base class.

        Args:
            methods: Optional method name or list of names to run. When None, methods
                will be auto-selected based on provided inputs.
        """
        if methods is None:
            self._user_methods = None
        elif isinstance(methods, str):
            self._user_methods = [methods]
        else:
            self._user_methods = methods
            
        # 存储已执行方法的结果
        self.results: Dict[str, Any] = {}
        
        # 跟踪已执行的方法
        self._executed_methods: List[str] = []
        
        # 存储最后使用的参数以供参考
        self._last_params: Dict[str, Any] = {}
    
    def _get_applicable_methods(self, **kwargs) -> List[str]:
        """Determine applicable methods from provided parameters.

        Args:
            **kwargs: User-provided parameters.

        Returns:
            List of method names that can be executed.
        """
        # 如果用户指定了方法，则验证并返回这些方法
        if self._user_methods:
            return self._validate_user_methods(self._user_methods, **kwargs)
        
        # 根据可用参数自动选择方法
        applicable_methods = []
        provided_params = set(k for k, v in kwargs.items() if v is not None)
        
        # 如果 _METHOD_MAP 可用（新方法），则使用它，否则回退到 _METHOD_REGISTRY
        method_source = getattr(self, '_METHOD_MAP', None) or self._METHOD_REGISTRY
        
        for method_name, method_info in method_source.items():
            if isinstance(method_info, list):
                # _METHOD_MAP 格式：方法名 -> [必需参数]
                required_params = set(method_info)
            else:
                # _METHOD_REGISTRY 格式：方法名 -> {'required': [...], ...}
                required_params = set(method_info.get('required', []))
            
            # 检查是否提供了所有必需的参数
            if required_params.issubset(provided_params):
                applicable_methods.append(method_name)
        
        if not applicable_methods:
            available_params = ', '.join(provided_params)
            available_methods = ', '.join(method_source.keys())
            raise ValueError(
                f"No applicable methods found for provided parameters: {available_params}. "
                f"Available methods: {available_methods}. "
                f"Please check the documentation for required parameters."
            )
        
        return applicable_methods
    
    def _validate_user_methods(self, methods: List[str], **kwargs) -> List[str]:
        """Validate user-specified methods against available parameters.

        Args:
            methods: Method names requested by the user.
            **kwargs: Available parameters.

        Returns:
            Valid method names.

        Raises:
            ValueError: When methods are unsupported or parameters are insufficient.
        """
        valid_methods = []
        provided_params = set(k for k, v in kwargs.items() if v is not None)
        
        # 如果 _METHOD_MAP 可用（新方法），则使用它，否则回退到 _METHOD_REGISTRY
        method_source = getattr(self, '_METHOD_MAP', None) or self._METHOD_REGISTRY
        
        for method in methods:
            if method not in method_source:
                raise ValueError(
                    f"Method '{method}' is not supported. "
                    f"Available methods: {', '.join(method_source.keys())}"
                )
            
            method_info = method_source[method]
            if isinstance(method_info, list):
                # _METHOD_MAP 格式：方法名 -> [必需参数]
                required_params = set(method_info)
            else:
                # _METHOD_REGISTRY 格式：方法名 -> {'required': [...], ...}
                required_params = set(method_info.get('required', []))
            
            missing_params = required_params - provided_params
            
            if missing_params:
                warnings.warn(
                    f"Method '{method}' requires missing parameters: {', '.join(missing_params)}. "
                    f"Skipping this method."
                )
                continue
            
            valid_methods.append(method)
        
        if not valid_methods:
            raise ValueError(
                "None of the specified methods can be executed with the provided parameters."
            )
        
        return valid_methods
    
    def decide(self, **kwargs) -> 'DecisionBase':
        """Run the decision methods.

        Common entrypoint; subclasses customize behavior via configuration and
        specialized logic.

        Args:
            **kwargs: Parameters for decision methods.

        Returns:
            Self, to allow chaining.
        """
        # 校验通用参数
        self._validate_common_parameters(**kwargs)
        
        # 调用子类的输入验证方法（如果存在）
        if hasattr(self, '_validate_inputs'):
            validated_params = self._validate_inputs(**kwargs)
            kwargs.update(validated_params)
        
        # 保存最近一次参数
        self._last_params = kwargs.copy()
        
        # 选择执行方法：优先严格校验用户指定，否则自动筛选
        if self._user_methods:
            # 用户指定了特定方法，使用严格验证
            methods_to_run = self._validate_user_methods(self._user_methods, **kwargs)
        else:
            # 自动选择方法
            methods_to_run = self._get_applicable_methods(**kwargs)
            if not methods_to_run:
                warnings.warn(f"No applicable {self.__class__.__name__} methods found for provided parameters")
                return self
        
        # 逐个执行方法并记录结果
        results_to_store = {}
        for method_name in methods_to_run:
            try:
                result = self._execute_method(method_name, **kwargs)
                self._store_result(method_name, result)
                results_to_store[method_name] = result
            except Exception as e:
                warnings.warn(f"Failed to execute {method_name}: {str(e)}")
        
        # 如果子类有ResultManager，则批量存储结果
        if hasattr(self, 'result_manager') and results_to_store:
            self._store_results_to_manager(results_to_store)
        
        return self
    
    @abstractmethod
    def _execute_method(self, method_name: str, **kwargs) -> Any:
        """Execute a specific method with given parameters.

        Subclasses must implement this method.

        Args:
            method_name: Name of the method to execute.
            **kwargs: Parameters for the method.

        Returns:
            Method result.
        """
        pass
    
    def _store_results_to_manager(self, results: Dict[str, Any]) -> None:
        """Store results into a ResultManager, if present.

        Subclasses may override to customize persistence behavior.

        Args:
            results: Dict mapping method names to their results.
        """
        if not hasattr(self, 'result_manager'):
            return
        
        # 根据子类类型选择相应的存储方法
        class_name = self.__class__.__name__.lower()
        
        if 'scoring' in class_name:
            self.result_manager.store_scoring_results(results)
        elif 'weighting' in class_name:
            self.result_manager.store_weighting_results(results)
        elif 'pairwise' in class_name:
            self.result_manager.store_pairwise_results(results)
        elif 'fuzzy' in class_name:
            self.result_manager.store_fuzzy_results(results)
        else:
            # 通用存储方法
            for method_name, result in results.items():
                self.result_manager.add_result(method_name, result)
    
    def get_results(self) -> Dict[str, Any]:
        """Return all executed method results as a dictionary."""
        return self.results.copy()
    
    def get_result(self, method_name: str) -> Optional[Any]:
        """Return the result for a particular method, or None if not executed."""
        return self.results.get(method_name)
    
    def get_executed_methods(self) -> List[str]:
        """Return a list of executed method names."""
        return self._executed_methods.copy()
    
    def clear_results(self) -> 'DecisionBase':
        """Clear stored results and state; returns self for chaining."""
        self.results.clear()
        self._executed_methods.clear()
        self._last_params.clear()
        return self
    
    def _store_result(self, method_name: str, result: Any) -> None:
        """Store a result for a given method name."""
        self.results[method_name] = result
        if method_name not in self._executed_methods:
            self._executed_methods.append(method_name)
    
    def _validate_common_parameters(self, **kwargs) -> None:
        """Validate common parameters.

        Raises:
            ValueError: Invalid values.
            TypeError: Wrong types.
        """
        # 如果提供了决策矩阵/数据集，则进行验证
        for dataset_key in ['decision_matrix', 'dataset']:
            if dataset_key in kwargs and kwargs[dataset_key] is not None:
                matrix = kwargs[dataset_key]
                if isinstance(matrix, str):
                    raise TypeError(f"{dataset_key} must be a numpy array, not string")
                if not isinstance(matrix, (np.ndarray, list)):
                    raise TypeError(f"{dataset_key} must be a numpy array or list")
                
                # 如果是列表，则转换为 numpy 数组
                if isinstance(matrix, list):
                    try:
                        matrix = np.array(matrix)
                    except Exception as e:
                        raise ValueError(f"Cannot convert {dataset_key} to numpy array: {e}")
                
                if matrix.ndim != 2:
                    raise ValueError(f"{dataset_key} must be 2-dimensional")
                if matrix.size == 0:
                    raise ValueError(f"{dataset_key} cannot be empty")
        
        # 如果提供了权重，则进行验证
        if 'weights' in kwargs and kwargs['weights'] is not None:
            weights = kwargs['weights']
            if not isinstance(weights, (np.ndarray, list)):
                raise TypeError("weights must be a numpy array or list")
            weights_array = np.array(weights) if isinstance(weights, list) else weights
            if weights_array.ndim != 1:
                raise ValueError("weights must be 1-dimensional")
            if np.any(weights_array < 0):
                raise ValueError("weights cannot contain negative values")
            if not np.isclose(np.sum(weights_array), 1.0, rtol=1e-3):
                warnings.warn("weights do not sum to 1.0, they will be normalized")
        
        # 如果提供了准则类型，则进行验证
        if 'criterion_type' in kwargs and kwargs['criterion_type'] is not None:
            criterion_type = kwargs['criterion_type']
            if not isinstance(criterion_type, list):
                raise TypeError("criterion_type must be a list")
            if len(criterion_type) == 0:
                raise ValueError("criterion_type cannot be empty")
            valid_types = {'max', 'min'}
            invalid_types = set(criterion_type) - valid_types
            if invalid_types:
                raise ValueError(
                    f"Invalid criterion types: {invalid_types}. "
                    f"Must be 'max' or 'min'"
                )
    
    def __repr__(self) -> str:
        """Debug-friendly string representation."""
        class_name = self.__class__.__name__
        n_executed = len(self._executed_methods)
        n_results = len(self.results)
        return f"{class_name}(executed_methods={n_executed}, results={n_results})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        if not self._executed_methods:
            return f"{self.__class__.__name__}: No methods executed yet"
        
        methods_str = ', '.join(self._executed_methods[:3])
        if len(self._executed_methods) > 3:
            methods_str += f", ... ({len(self._executed_methods)} total)"
        
        return f"{self.__class__.__name__}: Executed methods: {methods_str}"
