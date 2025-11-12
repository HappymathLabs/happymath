"""
模糊评分类决策方法。

本模块实现常见的基于评分/距离/效用的模糊排序方法，
通过综合模糊决策矩阵与权重、方向等信息生成方案得分与排名，
并提供统一的调度与结果访问接口。
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Any, Union

from ..algorithm import fuzzy_aras, fuzzy_copras, fuzzy_edas, fuzzy_moora, fuzzy_ocra, fuzzy_topsis, fuzzy_vikor, fuzzy_waspas

from ..core.base import DecisionBase
from ..core.method_registry import MethodRegistry
from ..core.validators import ParameterValidator
from ..core.utils import execute_algorithm_with_suppression, prepare_standard_algorithm_params
from ..results.result_manager import ResultManager


class FuzzyScoringDecision(DecisionBase):
    """
    模糊评分与排序方法集合。
    
    基于模糊决策矩阵和准则方向信息，计算各方案的综合得分并进行排序。
    
    必需参数:
        dataset: 模糊决策矩阵（备选方案 × 准则，每个元素为三角模糊数）
        criterion_type: 每个准则的方向（'max' 或 'min'）
    
    可选参数:
        weights: 准则权重向量（如未提供，将使用等权重或方法默认权重）
    """
    
    # 模糊评分方法注册表，用于统一筛选与调用
    _METHOD_REGISTRY = MethodRegistry.FUZZY_SCORING_METHODS
    
    def _validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """模糊评分方法输入参数校验与标准化。"""
        validated_params = {}
        
        # 提取关键参数
        dataset = kwargs.get('dataset')
        weights = kwargs.get('weights')
        criterion_type = kwargs.get('criterion_type')
        
        # 校验并标准化模糊决策矩阵
        if dataset is not None:
            validation_result = ParameterValidator.validate_fuzzy_matrix(dataset)
            if not validation_result['is_valid']:
                raise ValueError(validation_result['error_message'])
            validated_params['dataset'] = validation_result['processed_data']
        
        # 若提供权重，则进行维度与有效性校验/标准化
        if weights is not None:
            # 获取准则数量用于验证
            n_criteria = None
            if 'dataset' in validated_params:
                processed_dataset = validated_params['dataset']
                if hasattr(processed_dataset, 'shape') and len(processed_dataset.shape) >= 2:
                    n_criteria = processed_dataset.shape[1]
                elif isinstance(processed_dataset, list) and len(processed_dataset) > 0:
                    n_criteria = len(processed_dataset[0]) if hasattr(processed_dataset[0], '__len__') else None

            # 标准化模糊权重为算法所需形状：[[ (l,m,u), ... ]]
            std_weights = weights
            # 转换 ndarray 为原生 list，避免后续算法用索引时报错
            if isinstance(std_weights, np.ndarray):
                std_weights = std_weights.tolist()
            # 如果是 [(l,m,u), ...]，则外面包一层
            if isinstance(std_weights, list) and len(std_weights) > 0:
                first = std_weights[0]
                # 情况A：已是 [[(l,m,u), ...]]
                if isinstance(first, list) and len(first) > 0 and isinstance(first[0], (list, tuple, np.ndarray)) and len(first[0]) == 3:
                    pass
                # 情况B：是 [(l,m,u), ...]，需要包一层
                elif isinstance(first, (list, tuple, np.ndarray)) and len(first) == 3:
                    std_weights = [ [tuple(w) for w in std_weights] ]
                else:
                    # 非预期格式，直接原样传递，具体算法可能会报错，便于定位
                    pass

            validated_params['weights'] = std_weights
        
        # 若提供正负向，进行长度与取值校验
        if criterion_type is not None:
            # 获取准则数量用于验证
            n_criteria = None
            if 'dataset' in validated_params:
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
    
    def _validate_common_parameters(self, **kwargs) -> None:
        """
        重写基类的通用参数验证，跳过对模糊数据的标准验证。
        
        模糊数据集通常为 3D（备选×准则×3），模糊权重为二维（准则×3），
        因此不能沿用基类中的二维/一维硬校验。
        """
        # 创建不包含模糊数据的参数副本进行验证
        kwargs_for_validation = kwargs.copy()
        if 'dataset' in kwargs_for_validation:
            del kwargs_for_validation['dataset']
        if 'weights' in kwargs_for_validation:
            del kwargs_for_validation['weights']
        
        # 调用父类的验证方法
        super()._validate_common_parameters(**kwargs_for_validation)
    
    # 方法名称到必需参数列表的映射
    # 注意：dataset和criterion_type现在是decide()函数的必需参数，weights是可选参数
    _METHOD_MAP = {
        'fuzzy_aras': ['dataset', 'weights', 'criterion_type'],
        'fuzzy_copras': ['dataset', 'weights', 'criterion_type'],
        'fuzzy_edas': ['dataset', 'weights', 'criterion_type'],
        'fuzzy_moora': ['dataset', 'weights', 'criterion_type'],
        'fuzzy_ocra': ['dataset', 'weights', 'criterion_type'],
        'fuzzy_topsis': ['dataset', 'weights', 'criterion_type'],
        'fuzzy_vikor': ['dataset', 'weights', 'criterion_type'],
        'fuzzy_waspas': ['dataset', 'weights', 'criterion_type']
    }
    
    _ALGORITHM_MAP = {
        'fuzzy_aras': fuzzy_aras,
        'fuzzy_copras': fuzzy_copras,
        'fuzzy_edas': fuzzy_edas,
        'fuzzy_moora': fuzzy_moora,
        'fuzzy_ocra': fuzzy_ocra,
        'fuzzy_topsis': fuzzy_topsis,
        'fuzzy_vikor': fuzzy_vikor,
        'fuzzy_waspas': fuzzy_waspas
    }
    
    def __init__(self, methods: Optional[Union[str, List[str]]] = None):
        """
        初始化模糊评分方法集合。
        
        参数:
            methods: 指定执行的方法（字符串或列表），None 时自动选择可用方法。
        """
        super().__init__(methods)
        self.result_manager = ResultManager()
    
    def _execute_method(self, method_name: str, **kwargs) -> Any:
        """
        调度具体模糊评分方法。
        
        参数:
            method_name: 方法名
            **kwargs: 方法所需的全部参数（已经过验证和处理）
        返回:
            对应方法的运行结果
        """
        if method_name not in self._ALGORITHM_MAP:
            raise ValueError(f"Unknown fuzzy scoring method: {method_name}")
    
        # 获取所需参数
        required_params = self._METHOD_MAP.get(method_name, [])
        
        # 构造最终传递给算法的参数字典
        final_params = {
            'graph': False,
            'verbose': False
        }
        
        # 将必需参数加入
        final_params.update({k: kwargs[k] for k in required_params if k in kwargs})
        
        # 根据 `graph` 参数决定是否需要抑制绘图
        needs_plot_suppression = not kwargs.get('graph', False)

        # 执行算法
        algorithm_func = self._ALGORITHM_MAP[method_name]
        result = execute_algorithm_with_suppression(
            algorithm_func, 
            final_params, 
            needs_plot_suppression=needs_plot_suppression
        )
        
        return result

    def get_all_results(self) -> Dict[str, Any]:
        """
        获取所有结果的统一接口。
        
        Returns:
            包含所有结果的字典
        """
        return self.result_manager.get_all_results()
    
    # Result access methods
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
                # 多输出场景（如 Fuzzy VIKOR）
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
