"""
Scoring-based decision methods.

Implements common ranking methods based on scores/distances/utilities, combining
decision matrix with weights and criterion directions to produce scores and rankings.
Provides unified dispatch and result access.
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Any, Union

from ..algorithm import aras, borda, cocoso, codas, copras, copeland, cradis, edas, gra, lmaw, mabac, mairca, marcos, macbeth, mara, maut, moora, moosra, multimoora, ocra, oreste, piv, rafsi, regime, rov, saw, smart, spotis, topsis, todim, vikor, waspas, wisp, psi

from ..core.base import DecisionBase
from ..core.method_registry import MethodRegistry
from ..core.validators import ParameterValidator
from ..core.utils import execute_algorithm_with_suppression, filter_algorithm_params, prepare_standard_algorithm_params
from ..results.result_manager import ResultManager


class ScoringDecision(DecisionBase):
    """
    Collection of scoring and ranking methods.

    Required parameters:
        - dataset: Decision matrix (alternatives × criteria)
        - criterion_type: Direction for each criterion ('max' or 'min')

    Optional parameters:
        - weights: Criterion weight vector (defaults to equal or method-specific)
    """
    
    # 评分方法注册表，用于统一筛选与调用
    _METHOD_REGISTRY = MethodRegistry.SCORING_METHODS
    
    def _validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """Validate and normalize inputs for scoring methods."""
        validated_params = {}
        
        # Extract key parameters
        dataset = kwargs.get('dataset')
        weights = kwargs.get('weights')
        criterion_type = kwargs.get('criterion_type')
        
        # Validate and normalize decision matrix
        if dataset is not None:
            validation_result = ParameterValidator.validate_decision_matrix(dataset)
            if not validation_result['is_valid']:
                raise ValueError(validation_result['error_message'])
            validated_params['dataset'] = validation_result['processed_data']
        
        # Validate weights when provided
        if weights is not None:
            n_criteria = validated_params['dataset'].shape[1] if 'dataset' in validated_params else None
            weights_result = ParameterValidator.validate_weights(weights, n_criteria=n_criteria)
            if not weights_result['is_valid']:
                raise ValueError(weights_result['error_message'])
            validated_params['weights'] = weights_result['processed_data']
        
        # Validate criterion types when provided
        if criterion_type is not None:
            n_criteria = validated_params['dataset'].shape[1] if 'dataset' in validated_params else None
            criterion_type_result = ParameterValidator.validate_criterion_type(
                criterion_type, n_criteria=n_criteria
            )
            if not criterion_type_result['is_valid']:
                raise ValueError(criterion_type_result['error_message'])
            validated_params['criterion_type'] = criterion_type_result['processed_data']
        
        return validated_params
    
    # Mapping from method name to required parameters
    _METHOD_MAP = {
        'topsis': ['dataset', 'criterion_type'],
        'vikor': ['dataset', 'criterion_type'],
        'saw': ['dataset', 'criterion_type'],
        'aras': ['dataset', 'criterion_type'],
        'copras': ['dataset', 'criterion_type'],
        'edas': ['dataset', 'criterion_type'],
        'codas': ['dataset', 'criterion_type'],
        'cocoso': ['dataset', 'criterion_type'],
        'cradis': ['dataset', 'criterion_type'],
        'mabac': ['dataset', 'criterion_type'],
        'mairca': ['dataset', 'criterion_type'],
        'marcos': ['dataset', 'criterion_type'],
        'moora': ['dataset', 'criterion_type'],
        'moosra': ['dataset', 'criterion_type'],
        'multimoora': ['dataset', 'criterion_type'],
        'waspas': ['dataset', 'criterion_type', 'lambda_value'],
        'wisp': ['dataset', 'criterion_type'],
        'todim': ['dataset', 'criterion_type'],
        'gra': ['dataset', 'criterion_type'],
        'lmaw': ['dataset', 'criterion_type'],
        'rafsi': ['dataset', 'criterion_type'],
        'spotis': ['dataset', 'criterion_type', 's_min', 's_max'],
        'macbeth': ['dataset', 'criterion_type'],
        'mara': ['dataset', 'criterion_type'],
        'ocra': ['dataset', 'criterion_type'],
        'oreste': ['dataset', 'criterion_type'],
        'piv': ['dataset', 'criterion_type'],
        'rov': ['dataset', 'criterion_type'],
        'borda': ['dataset', 'criterion_type'],
        'copeland': ['dataset', 'criterion_type'],
        'smart': ['dataset', 'criterion_type', 'grades', 'lower', 'upper'],
        'maut': ['dataset', 'criterion_type', 'utility_functions'],
        'psi': ['dataset', 'criterion_type']
    }
    
    _ALGORITHM_MAP = {
        'topsis': topsis,
        'vikor': vikor,
        'saw': saw,
        'aras': aras,
        'copras': copras,
        'edas': edas,
        'codas': codas,
        'cocoso': cocoso,
        'cradis': cradis,
        'mabac': mabac,
        'mairca': mairca,
        'marcos': marcos,
        'moora': moora,
        'moosra': moosra,
        'multimoora': multimoora,
        'waspas': waspas,
        'wisp': wisp,
        'todim': todim,
        'gra': gra,
        'lmaw': lmaw,
        'rafsi': rafsi,
        'spotis': spotis,
        'macbeth': macbeth,
        'mara': mara,
        'ocra': ocra,
        'oreste': oreste,
        'piv': piv,
        'rov': rov,
        'borda': borda,
        'copeland': copeland,
        'smart': smart,
        'maut': maut,
        'psi': psi
    }
    
    def __init__(self, methods: Optional[Union[str, List[str]]] = None):
        """
        初始化评分方法集合。
        
        参数:
            methods: 指定执行的方法（字符串或列表），None 时自动选择可用方法。
        """
        super().__init__(methods)
        self.result_manager = ResultManager()
    
    def _execute_method(self, method_name: str, **kwargs) -> Any:
        """
        调度具体评分方法。
        
        参数:
            method_name: 方法名
            **kwargs: 方法所需的全部参数（已经过验证和处理）
        返回:
            对应方法的运行结果
        """
        if method_name not in self._ALGORITHM_MAP:
            raise ValueError(f"Unknown scoring method: {method_name}")
    
        # 获取所需参数
        required_params = self._METHOD_MAP.get(method_name, [])
        
        # 构造最终传递给算法的参数字典
        final_params = {
            'graph': False,
            'verbose': False
        }
        
        # 将必需参数加入
        final_params.update({k: kwargs[k] for k in required_params if k in kwargs})
        
        # 将可选的weights参数加入（如果提供的话）
        if 'weights' in kwargs and kwargs['weights'] is not None:
            final_params['weights'] = kwargs['weights']
        
        # 特殊参数处理
        if 's_min' in kwargs:
            final_params['s_min'] = np.array(kwargs['s_min'])
        if 's_max' in kwargs:
            final_params['s_max'] = np.array(kwargs['s_max'])

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
    
    def get_scores(self, method: Optional[str] = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        获取各方法计算得到的方案得分向量。
        
        参数:
            method: 指定方法名，None 时返回所有方法的字典
        返回:
            单个得分向量（或方法名到得分的映射）
        """
        if method:
            result = self.results.get(method)
            if result is None:
                raise ValueError(f"No results found for method: {method}")
            
            # 从不同方法的返回中提取得分
            if isinstance(result, np.ndarray):
                if result.ndim == 2 and result.shape[1] == 2:
                    # 流格式 [id, score]
                    return result[:, 1]
                else:
                    return result
            elif isinstance(result, tuple):
                # 多输出场景（如 VIKOR）
                return result
            else:
                return result
        else:
            # 返回所有方法的得分
            return self.result_manager.get_all_scores()
    
    def get_rankings(self, method: Optional[str] = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        获取排名结果。
        
        参数:
            method: 指定方法名，None 时返回所有方法的字典
        返回:
            排名向量（或方法名到排名的映射）
        """
        if method:
            scores = self.get_scores(method)
            if scores is None:
                raise ValueError(f"No scores found for method: {method}")
            
            # 将得分转换为排名（高分更优）
            if isinstance(scores, np.ndarray):
                if scores.ndim == 1:
                    # Higher score = better rank
                    return np.argsort(-scores) + 1
                elif scores.ndim == 2 and scores.shape[1] == 2:
                    # 已为包含排名的流格式
                    return scores[:, 0].astype(int)
            
            return scores
        else:
            # 返回所有方法的排名
            return self.result_manager.get_all_rankings()
    
    def compare_rankings(self) -> Any:
        """
        比较不同方法的排名结果（返回 DataFrame）。
        """
        return self.result_manager.compare_rankings()
    
    def compare_scores(self) -> Any:
        """
        比较不同方法的得分结果（返回 DataFrame）。
        """
        return self.result_manager.compare_scores()
