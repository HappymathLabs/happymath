"""
成对比较类决策方法。

本模块实现 ELECTRE 与 PROMETHEE 两大系列的外排序/优势关系构建方法，
用于在多方案、多准则情境下通过成对比较建立优势关系矩阵与流值等指标。
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Any, Union

from ..algorithm import e_iii, e_iv, p_ii

from ..core.base import DecisionBase
from ..core.method_registry import MethodRegistry
from ..core.validators import ParameterValidator
from ..core.utils import execute_algorithm_with_suppression, prepare_standard_algorithm_params
from ..results.result_manager import ResultManager


class PairwiseDecision(DecisionBase):
    """
    成对比较方法集合（ELECTRE 与 PROMETHEE 家族）。
    
    这些方法通过构造一致性、反一致性、偏好函数等，
    在候选方案之间建立优势/支配关系，并给出核集、净流等结果。
    
    必需参数:
        dataset: 决策矩阵（备选方案 × 准则）
        weights: 准则权重向量
        
    可选参数:
        P: 偏好阈值
        Q: 无差异阈值  
        V: 否决阈值
        
    特殊说明:
        - ELECTRE IV 方法不使用权重参数
        - PROMETHEE VI 方法需要 W_lower 和 W_upper 参数而非标准权重
        
    使用示例:
        # 基本使用
        pairwise = PairwiseDecision()
        pairwise.decide(dataset=data_matrix, weights=weight_vector)
        
        # 使用阈值参数
        pairwise.decide(dataset=data_matrix, weights=weight_vector,
                       P=preference_thresholds, Q=indifference_thresholds)
        
        # 指定特定方法
        pairwise = PairwiseDecision(['electre_i', 'promethee_ii'])
        pairwise.decide(dataset=data_matrix, weights=weight_vector)
        
        # PROMETHEE VI 特殊用法
        pairwise = PairwiseDecision(['promethee_vi'])
        pairwise.decide(dataset=data_matrix, weights=weight_vector,
                       W_lower=lower_weights, W_upper=upper_weights)
    """
    
    # 成对比较方法的注册表，供统一筛选与调用
    _METHOD_REGISTRY = MethodRegistry.PAIRWISE_METHODS
    
    # 方法名称到必需参数列表的映射，用于自动选择与严格校验
    # 注意：dataset和weights现在是decide()函数的必需参数，特殊方法会在执行时处理参数差异
    _METHOD_MAP = {
        # ELECTRE methods (根据真实函数签名)
        'electre_iii': ['dataset', 'weights', 'P', 'Q', 'V'],
        'electre_iv': ['dataset', 'P', 'Q', 'V'],
        
        # PROMETHEE methods (根据真实函数签名)
        'promethee_ii': ['dataset', 'weights', 'Q', 'S', 'P', 'F']
    }
    
    _ALGORITHM_MAP = {
        'electre_iii': e_iii.electre_iii,
        'electre_iv': e_iv.electre_iv,
        'promethee_ii': p_ii.promethee_ii
    }
    
    def __init__(self, methods: Optional[Union[str, List[str]]] = None):
        """
        初始化成对比较方法集合。
        
        参数:
            methods: 指定要执行的方法（字符串或列表）。为 None 时按输入自动选择。
        """
        super().__init__(methods)
        self.result_manager = ResultManager()
    
    def _validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """成对比较方法输入参数校验与标准化。"""
        validated_params = {}
        
        # 提取关键参数
        dataset = kwargs.get('dataset')
        weights = kwargs.get('weights')
        
        # 校验并标准化决策矩阵
        if dataset is not None:
            validation_result = ParameterValidator.validate_decision_matrix(dataset)
            if not validation_result['is_valid']:
                raise ValueError(validation_result['error_message'])
            validated_params['dataset'] = validation_result['processed_data']
            n_criteria = validated_params['dataset'].shape[1]
        else:
            n_criteria = None
        
        if weights is not None:
            weight_result = ParameterValidator.validate_weights(weights, n_criteria=n_criteria)
            if not weight_result['is_valid']:
                raise ValueError(weight_result['error_message'])
            validated_params['weights'] = weight_result['processed_data']
        
        # 阈值参数处理
        for key in ['P', 'Q', 'V', 'S']:
            if key in kwargs and kwargs[key] is not None:
                validated_params[key] = np.array(kwargs[key]) if not isinstance(kwargs[key], np.ndarray) else kwargs[key]
        
        # ELECTRE 阈值校验
        if any(k in kwargs for k in ['P', 'Q', 'V']):
            threshold_result = ParameterValidator.validate_thresholds(
                kwargs.get('P'), kwargs.get('Q'), kwargs.get('V'), n_criteria=n_criteria
            )
            if not threshold_result['is_valid']:
                raise ValueError(threshold_result['error_message'])
            
            p_val, q_val, v_val = threshold_result['processed_data']
            if p_val is not None: validated_params['P'] = p_val
            if q_val is not None: validated_params['Q'] = q_val
            if v_val is not None: validated_params['V'] = v_val
        
        # 其他参数处理
        if 'F' in kwargs and kwargs['F'] is not None:
            validated_params['F'] = np.array(kwargs['F']) if not isinstance(kwargs['F'], np.ndarray) else kwargs['F']
        
        return validated_params
    
    def _handle_special_method_params(self, method_name: str, **kwargs) -> Dict[str, Any]:
        """
        处理特殊方法的参数需求。
        
        参数:
            method_name: 方法名称
            **kwargs: 原始参数
            
        返回:
            处理后的参数字典
        """
        method_kwargs = kwargs.copy()
        
        # ELECTRE IV 不需要权重参数
        if method_name == 'electre_iv':
            # 移除权重参数，因为该方法不使用权重
            method_kwargs.pop('W', None)
        
        return method_kwargs
    
    def _execute_method(self, method_name: str, **kwargs) -> Any:
        """
        调度具体成对比较方法。
        
        参数:
            method_name: 方法名称
            **kwargs: 对应方法的参数（已经过验证和特殊处理）
        返回:
            方法执行结果
        """
        if method_name not in self._ALGORITHM_MAP:
            raise ValueError(f"Unknown pairwise method: {method_name}")

        algorithm_func = self._ALGORITHM_MAP[method_name]
        
        # 处理特殊方法的参数需求
        method_kwargs = self._handle_special_method_params(method_name, **kwargs)

        # 构造最终传递给算法的参数字典
        final_params = prepare_standard_algorithm_params(
            sort=True,  # for promethee
            topn=0      # for promethee
        )
        
        # 获取所需参数并添加到最终参数中
        required_params = self._METHOD_MAP.get(method_name, [])
        final_params.update({k: method_kwargs[k] for k in required_params if k in method_kwargs})
        
        # 添加其他可选参数
        final_params.update({k: v for k, v in method_kwargs.items() 
                           if k not in required_params and v is not None})

        # 执行算法
        result = execute_algorithm_with_suppression(
            algorithm_func, 
            final_params, 
            False  # 不需要绘图抑制
        )
        
        return result
    
    def get_all_results(self) -> Dict[str, Any]:
        """
        获取所有结果的统一接口。
        
        Returns:
            包含所有结果的字典
        """
        return self.result_manager.get_all_results()
    
    def get_kernel(self, method: str) -> Optional[set]:
        """
        获取 ELECTRE 方法得到的核集（非支配解集合）。
        
        参数:
            method: 方法名称（如 'electre_i'）
        返回:
            非支配方案集合 set，若不可用返回 None
        """
        result = self.results.get(method)
        if result is None:
            return None
        
        # ELECTRE I 返回 (一致性, 反一致性, 支配矩阵, 核集, 被支配集)
        if isinstance(result, tuple) and len(result) >= 4:
            return result[3]  # kernel is the 4th element
        
        return None
    
    def get_outranking_matrix(self, method: str) -> Optional[np.ndarray]:
        """
        获取优势/支配矩阵。
        
        参数:
            method: 方法名称
        返回:
            优势/支配矩阵（numpy 数组），不可用则返回 None
        """
        result = self.results.get(method)
        if result is None:
            return None
        
        # ELECTRE I 中，支配矩阵位于返回元组的第三项
        if isinstance(result, tuple) and len(result) >= 3:
            return result[2]  # dominance matrix
        
        # PROMETHEE I 直接返回比较矩阵
        if isinstance(result, np.ndarray):
            return result
        
        return None
    
    def get_net_flows(self, method: str = 'promethee_ii') -> Optional[np.ndarray]:
        """
        获取 PROMETHEE II 的净流值。
        
        参数:
            method: 方法名称（默认 'promethee_ii'）
        返回:
            净流值数组，若不可用返回 None
        """
        result = self.results.get(method)
        if result is None:
            return None
        
        # PROMETHEE II 返回形状为 (n, 2) 的数组，其中第二列是净流值
        if isinstance(result, np.ndarray) and result.ndim == 2 and result.shape[1] >= 2:
            return result[:, 1]  # 返回第二列（净流值）
        
        return None
    
    def compare_rankings(self) -> Any:
        """
        比较不同方法的排名结果（返回 DataFrame）。
        """
        return self.result_manager.compare_rankings()
