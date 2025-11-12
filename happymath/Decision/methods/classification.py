"""
方案分类决策方法。

本模块实现 ELECTRE TRI-B 与 CPP Tri 等方法，
用于将备选方案分配到预定义的类别中。
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union

from ..algorithm import e_tri_b, cpp_tri
from ..core.base import DecisionBase
from ..core.method_registry import MethodRegistry
from ..core.validators import ParameterValidator
from ..core.utils import execute_algorithm_with_suppression
from ..results.result_manager import ResultManager


class ClsDecision(DecisionBase):
    """
    方案分类方法集合。
    
    根据备选方案在各准则下的表现及预设的分类边界，将其分配到不同类别。
    """
    
    _METHOD_REGISTRY = MethodRegistry.CLASSIFICATION_METHODS
    
    _METHOD_MAP = {
        'electre_tri_b': ['dataset', 'weights', 'Q', 'P', 'V', 'B'],
        'cpp_tri': ['dataset']  # cpp_tri 接受 dataset 作为决策矩阵，weights 可选，profiles/num_cat 可定义分类边界
    }
    
    _ALGORITHM_MAP = {
        'electre_tri_b': e_tri_b,
        'cpp_tri': cpp_tri
    }

    def __init__(self, methods: Optional[Union[str, List[str]]] = None):
        super().__init__(methods)
        self.result_manager = ResultManager()

    def _validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """输入参数校验与标准化。"""
        validated_params = {}
        dataset = kwargs.get('dataset')
        weights = kwargs.get('weights')

        # 校验决策矩阵
        if dataset is not None:
            res = ParameterValidator.validate_decision_matrix(dataset)
            if not res['is_valid']: 
                raise ValueError(res['error_message'])
            validated_params['dataset'] = res['processed_data']
            n_criteria = validated_params['dataset'].shape[1]
        else:
            n_criteria = None

        if weights is not None:
            res = ParameterValidator.validate_weights(weights, n_criteria=n_criteria)
            if not res['is_valid']: 
                raise ValueError(res['error_message'])
            validated_params['weights'] = res['processed_data']

        # 阈值参数 P, Q, V
        if any(k in kwargs for k in ['P', 'Q', 'V']):
            res = ParameterValidator.validate_thresholds(
                kwargs.get('P'), kwargs.get('Q'), kwargs.get('V'), n_criteria=n_criteria
            )
            if not res['is_valid']: 
                raise ValueError(res['error_message'])
            p, q, v = res['processed_data']
            if p is not None: validated_params['P'] = p
            if q is not None: validated_params['Q'] = q
            if v is not None: validated_params['V'] = v
        
        # 分类边界 B 和 profiles 以及其他可选参数
        for key in ['B', 'profiles', 'num_cat', 'cut_level', 'rule', 'indep_criteria', 'verbose']:
            if key in kwargs and kwargs[key] is not None:
                validated_params[key] = kwargs[key]
        
        return validated_params

    def _execute_method(self, method_name: str, **kwargs) -> Any:
        """调度并执行分类算法。"""
        algorithm_func = self._ALGORITHM_MAP[method_name]
        
        # 参数名映射与准备
        params = kwargs.copy()
        
        # 构造最终传递给算法的参数字典
        final_params = {'verbose': False}
        
        # 获取所需参数并添加到最终参数中
        required_params = self._METHOD_MAP.get(method_name, [])
        final_params.update({k: params[k] for k in required_params if k in params})
        
        # 添加其他可选参数
        final_params.update({k: v for k, v in params.items() 
                           if k not in required_params and v is not None})

        return execute_algorithm_with_suppression(algorithm_func, final_params)

    def get_classification(self, method: str) -> Optional[List[int]]:
        """获取指定方法的分类结果。"""
        return self.results.get(method)
        
    def get_all_results(self) -> Dict[str, Any]:
        """获取所有结果。"""
        return self.result_manager.get_all_results()

    def compare_classifications(self, methods: Optional[List[str]] = None):
        """比较不同方法的分类结果。"""
        return self.result_manager.compare_classifications(methods)
