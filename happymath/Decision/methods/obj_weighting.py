"""
客观权重确定方法集合。

本模块实现基于数据结构的权重计算方法（如 CRITIC/Entropy/MEREC/PSI 等），
统一封装输入校验、方法调度与结果访问接口，便于在多准则场景中获取基于数据驱动的准则权重。
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
    客观权重方法集合。
    
    基于原始决策矩阵的权重确定方法，通过分析数据结构来确定准则权重。
    提供统一的调用与结果比较、聚合能力。
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
        初始化客观权重方法集合。
        
        参数:
            methods: 指定要执行的方法（字符串或列表）。为 None 时自动选择可用方法。
        """
        super().__init__(methods)
        self.result_manager = ResultManager()
    
    def _validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """客观权重方法输入参数校验与标准化。"""
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
        调度并执行具体的客观权重计算方法。
        
        参数:
            method_name: 方法名称
            **kwargs: 该方法的入参（已经过校验）
        返回:
            方法执行结果
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
        """
        获取所有结果的统一接口。
        
        Returns:
            包含所有结果的字典
        """
        return self.result_manager.get_all_results()
    
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
    
