"""
所有决策方法的抽象基类。

本模块为所有决策分析类提供了基础，
包括通用功能和接口。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import numpy as np
import warnings


class DecisionBase(ABC):
    """
    所有决策方法类的抽象基类。
    
    该类提供：
    - 所有决策方法的通用接口
    - 基于可用参数的智能方法选择
    - 结果存储和管理
    - 参数验证框架
    """
    
    # 方法注册表：将方法名称映射到其所需参数
    # 子类应重写此属性
    _METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {}
    
    def __init__(self, methods: Optional[Union[str, List[str]]] = None):
        """
        初始化决策基类。
        
        参数:
            methods: 要使用的特定方法。如果为 None，将根据参数自动选择。
                     可以是单个方法名（str）或方法名列表。
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
        """
        根据提供的参数确定可以执行哪些方法。
        
        参数:
            **kwargs: 用户提供的参数
            
        返回:
            可以执行的方法名称列表
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
        """
        根据可用参数验证用户指定的方法。
        
        参数:
            methods: 用户指定的方法名称列表
            **kwargs: 可用的参数
            
        返回:
            有效的方法名称列表
            
        引发:
            ValueError: 如果方法不受支持或参数不足
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
        """
        执行决策方法。
        
        这是所有决策分析的主要入口点。
        通用的决策执行流程，子类通过提供配置和特定逻辑来定制行为。
        
        参数:
            **kwargs: 决策方法的参数
            
        返回:
            返回自身以支持方法链式调用
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
        """
        使用给定参数执行特定方法。
        
        子类必须实现此方法以处理其特定方法。
        
        参数:
            method_name: 要执行的方法名称
            **kwargs: 方法的参数
            
        返回:
            方法执行的结果
        """
        pass
    
    def _store_results_to_manager(self, results: Dict[str, Any]) -> None:
        """
        将结果存储到ResultManager中。
        
        子类可以重写此方法来定制存储行为。
        默认实现会根据子类类型自动选择合适的存储方法。
        
        参数:
            results: 要存储的结果字典 {method_name: result}
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
        """
        获取所有已执行方法的结果。
        
        返回:
            一个将方法名称映射到其结果的字典
        """
        return self.results.copy()
    
    def get_result(self, method_name: str) -> Optional[Any]:
        """
        获取特定方法的结果。
        
        参数:
            method_name: 方法的名称
            
        返回:
            方法的结果，如果尚未执行则返回 None
        """
        return self.results.get(method_name)
    
    def get_executed_methods(self) -> List[str]:
        """
        获取已执行的方法列表。
        
        返回:
            已执行的方法名称列表
        """
        return self._executed_methods.copy()
    
    def clear_results(self) -> 'DecisionBase':
        """
        清除所有存储的结果。
        
        返回:
            返回自身以支持方法链式调用
        """
        self.results.clear()
        self._executed_methods.clear()
        self._last_params.clear()
        return self
    
    def _store_result(self, method_name: str, result: Any) -> None:
        """
        存储方法执行的结果。
        
        参数:
            method_name: 方法的名称
            result: 要存储的结果
        """
        self.results[method_name] = result
        if method_name not in self._executed_methods:
            self._executed_methods.append(method_name)
    
    def _validate_common_parameters(self, **kwargs) -> None:
        """
        验证许多方法通用的参数。
        
        参数:
            **kwargs: 要验证的参数
            
        引发:
            ValueError: 如果参数无效
            TypeError: 如果参数类型错误
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
        """对象的字符串表示形式。"""
        class_name = self.__class__.__name__
        n_executed = len(self._executed_methods)
        n_results = len(self.results)
        return f"{class_name}(executed_methods={n_executed}, results={n_results})"
    
    def __str__(self) -> str:
        """人类可读的字符串表示形式。"""
        if not self._executed_methods:
            return f"{self.__class__.__name__}: No methods executed yet"
        
        methods_str = ', '.join(self._executed_methods[:3])
        if len(self._executed_methods) > 3:
            methods_str += f", ... ({len(self._executed_methods)} total)"
        
        return f"{self.__class__.__name__}: Executed methods: {methods_str}"
