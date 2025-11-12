"""
模糊客观权重确定方法集合。

本模块实现基于数据结构的模糊权重计算方法（如 Fuzzy CRITIC/MEREC 等），
统一封装输入校验、方法调度与结果访问接口，便于在多准则场景中获取基于数据驱动的模糊准则权重。
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Any, Union

from ..algorithm import fuzzy_critic, fuzzy_merec

from ..core.base import DecisionBase
from ..core.method_registry import MethodRegistry
from ..core.validators import ParameterValidator
from ..core.utils import execute_algorithm_with_suppression, prepare_standard_algorithm_params
from ..results.result_manager import ResultManager


class FuzzyObjWeighting(DecisionBase):
    """
    模糊客观权重方法集合。
    
    基于原始模糊决策矩阵的权重确定方法，通过分析数据结构来确定准则权重。
    提供统一的调用与结果比较、聚合能力。
    """
    
    # 模糊客观权重方法的注册表
    _METHOD_REGISTRY = MethodRegistry.FUZZY_OBJ_WEIGHTING_METHODS
    
    # 方法到其必需参数的映射
    _METHOD_MAP = {
        'fuzzy_critic': ['dataset', 'criterion_type'],
        'fuzzy_merec': ['dataset', 'criterion_type']
    }
    
    _ALGORITHM_MAP = {
        'fuzzy_critic': fuzzy_critic,
        'fuzzy_merec': fuzzy_merec
    }
    
    def __init__(self, methods: Optional[Union[str, List[str]]] = None):
        """
        初始化模糊客观权重方法集合。
        
        参数:
            methods: 指定要执行的方法（字符串或列表）。为 None 时自动选择可用方法。
        """
        super().__init__(methods)
        self.result_manager = ResultManager()

    def decide(self, dataset, criterion_type, **kwargs) -> 'FuzzyObjWeighting':
        """
        按传入参数执行模糊客观权重计算方法。
        
        必需参数:
            dataset: 模糊决策矩阵（方案×准则，每个元素为三角模糊数）
            criterion_type: 每个准则的方向（'max' 或 'min'）
            
        可选参数:
            **kwargs: 其他特定方法需要的参数
            
        返回:
            返回 self 以便链式调用
        """
        # 校验必需参数
        validated_params = self._validate_standard_inputs(dataset, criterion_type)
        
        # 校验其他可选参数
        self._validate_common_parameters(**kwargs)
        
        # 构建完整的参数字典
        all_params = validated_params.copy()
        all_params.update(kwargs)
        
        # 保存最近一次参数
        self._last_params = all_params.copy()
        
        # 选择执行方法：优先严格校验用户显式指定，否则自动筛选
        if self._user_methods:
            # 用户指定了特定方法，使用严格验证
            methods_to_run = self._validate_user_methods(self._user_methods, **all_params)
        else:
            # 自动选择方法
            methods_to_run = self._get_applicable_methods(**all_params)
            if not methods_to_run:
                warnings.warn("No applicable fuzzy objective weighting methods found for provided parameters")
                return self
        
        # 逐个执行并记录结果
        results_to_store = {}
        for method_name in methods_to_run:
            try:
                result = self._execute_method(method_name, **all_params)
                self._store_result(method_name, result)
                results_to_store[method_name] = result
            except Exception as e:
                warnings.warn(f"Failed to execute {method_name}: {str(e)}")
        
        # 使用专门的模糊权重存储方法
        if results_to_store:
            self.result_manager.store_fuzzy_results(results_to_store)
        
        return self
    
    def _execute_method(self, method_name: str, **kwargs) -> Any:
        """
        调度并执行具体的模糊客观权重计算方法。
        
        参数:
            method_name: 方法名称
            **kwargs: 该方法的入参（已经过校验）
        返回:
            方法执行结果
        """
        if method_name not in self._ALGORITHM_MAP:
            raise ValueError(f"Unknown fuzzy objective weighting method: {method_name}")

        # 构造最终传递给算法的参数字典
        final_params = prepare_standard_algorithm_params()
        
        # 获取所需参数并添加到最终参数中
        required_params = self._METHOD_MAP.get(method_name, [])
        final_params.update({k: kwargs[k] for k in required_params if k in kwargs})

        # 执行算法
        algorithm_func = self._ALGORITHM_MAP[method_name]
        result = execute_algorithm_with_suppression(algorithm_func, final_params)
        
        return result

    def _validate_standard_inputs(self, dataset, criterion_type):
        """模糊客观权重方法必需参数校验与标准化。"""
        validated_params = {}
        
        # 校验并标准化模糊决策矩阵
        validation_result = ParameterValidator.validate_fuzzy_matrix(dataset)
        if not validation_result['is_valid']:
            raise ValueError(validation_result['error_message'])
        validated_params['dataset'] = validation_result['processed_data']
        
        # 校验准则类型
        # 获取准则数量用于验证
        n_criteria = None
        processed_dataset = validated_params['dataset']
        if hasattr(processed_dataset, 'shape') and len(processed_dataset.shape) >= 2:
            n_criteria = processed_dataset.shape[1]
        elif isinstance(processed_dataset, list) and len(processed_dataset) > 0:
            n_criteria = len(processed_dataset[0]) if hasattr(processed_dataset[0], '__len__') else None
            
        criterion_type_result = ParameterValidator.validate_criterion_type(
            criterion_type, n_criteria=n_criteria
        )
        if not criterion_type_result['is_valid']:
            raise ValueError(criterion_type_result['error_message'])
        validated_params['criterion_type'] = criterion_type_result['processed_data']
        
        return validated_params

    def get_all_results(self) -> Dict[str, Any]:
        """
        获取所有结果的统一接口。
        
        Returns:
            包含所有结果的字典
        """
        return self.result_manager.get_all_results()

    def get_fuzzy_weights(self, method: Optional[str] = None) -> Union[Any, Dict[str, Any]]:
        """
        获取已执行方法的模糊权重。
        
        参数:
            method: 指定方法名；为 None 时返回全部模糊权重的字典
            
        返回:
            指定方法的模糊权重，或方法名到模糊权重的映射字典
        """
        if method:
            result = self.results.get(method)
            if result is None:
                raise ValueError(f"No results found for method: {method}")
            
            return result
        else:
            # 返回所有模糊方法的权重
            all_weights = {}
            for name, result in self.results.items():
                if 'fuzzy' in name.lower():
                    all_weights[name] = result
            return all_weights
    
    def get_weights(self, method: Optional[str] = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        获取各方法计算得到的权重向量。
        
        参数:
            method: 指定方法名；None 时返回所有方法的权重字典
        返回:
            单个权重向量或方法名到权重的映射
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
        """
        跨方法比较权重（返回 DataFrame）。
        """
        return self.result_manager.compare_weights()
    
