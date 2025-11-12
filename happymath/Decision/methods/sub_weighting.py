"""
主观权重确定方法集合。

本模块实现基于专家判断的权重计算方法（如 AHP/BWM/FUCOM/ROC 等），
统一封装输入校验、方法调度与结果访问接口，便于在多准则场景中获取基于主观判断的准则权重。
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Any, Union

from ..algorithm import ahp, bwm, bwm_s, fucom, roc, rrw, rsw, dematel, wings

from ..core.base import DecisionBase
from ..core.method_registry import MethodRegistry
from ..core.validators import ParameterValidator
from ..core.utils import execute_algorithm_with_suppression
from ..results.result_manager import ResultManager


class SubWeighting(DecisionBase):
    """
    主观权重方法集合。
    
    基于专家判断的权重确定方法，如成对比较或直接排序。
    提供统一的调用与结果比较、聚合能力。
    """
    
    # 主观权重方法的注册表
    _METHOD_REGISTRY = MethodRegistry.SUB_WEIGHTING_METHODS
    
    # 方法到其必需参数的映射
    _METHOD_MAP = {
        'ahp': ['dataset'],  # 成对比较矩阵
        'bwm': ['mic', 'lic'],  # 最佳到其他、其他到最差向量
        'simplified_bwm': ['mic', 'lic'],  # 简化BWM
        'fucom': ['criteria_rank', 'criteria_priority'],  # 排序和优先级
        'roc': ['criteria_rank'],  # 排序
        'rrw': ['criteria_rank'],  # 排序
        'rsw': ['criteria_rank'],   # 排序
        'dematel': ['dataset'],  # 直接影响关系矩阵
        'wings': ['dataset']     # 直接影响关系矩阵
    }
    
    _ALGORITHM_MAP = {
        'ahp': ahp,
        'bwm': bwm.bw,
        'simplified_bwm': bwm_s.simplified_bw,
        'fucom': fucom,
        'roc': roc,
        'rrw': rrw,
        'rsw': rsw,
        'dematel': dematel,
        'wings': wings
    }

    def __init__(self, methods: Optional[Union[str, List[str]]] = None):
        """
        初始化主观权重方法集合。
        
        参数:
            methods: 指定要执行的方法（字符串或列表）。为 None 时自动选择可用方法。
        """
        super().__init__(methods)
        self.result_manager = ResultManager()
    
    def _validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """主观权重方法输入参数校验与标准化。"""
        validated_params = {}
        
        # 根据可能的方法进行参数校验
        # 由于主观权重方法参数各异，我们根据提供的参数类型进行校验
        
        # AHP方法参数校验
        if 'dataset' in kwargs and kwargs['dataset'] is not None:
            val_res = ParameterValidator.validate_pairwise_matrix(kwargs['dataset'])
            if not val_res['is_valid']:
                raise ValueError(val_res['error_message'])
            validated_params['dataset'] = val_res['processed_data']
        
        # BWM方法参数校验
        if 'mic' in kwargs and kwargs['mic'] is not None:
            validated_params['mic'] = np.array(kwargs['mic'], dtype=float)
        if 'lic' in kwargs and kwargs['lic'] is not None:
            validated_params['lic'] = np.array(kwargs['lic'], dtype=float)
        
        return validated_params
    
    def _execute_method(self, method_name: str, **kwargs) -> Any:
        """
        调度并执行具体的主观权重计算方法。
        
        参数:
            method_name: 方法名称
            **kwargs: 该方法的入参（已经过校验）
        返回:
            方法执行结果
        """
        if method_name not in self._ALGORITHM_MAP:
            raise ValueError(f"Unknown subjective weighting method: {method_name}")

        algorithm_func = self._ALGORITHM_MAP[method_name]
        
        # 构造最终传递给算法的参数字典
        final_params = {'verbose': False}
        
        # 获取所需参数并添加到最终参数中
        required_params = self._METHOD_MAP.get(method_name, [])
        final_params.update({k: kwargs[k] for k in required_params if k in kwargs})
        
        # 添加其他可选参数
        final_params.update({k: v for k, v in kwargs.items() 
                           if k not in required_params and v is not None})

        # 执行算法
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
                if method in ['dematel', 'wings']:
                    # DEMATEL和WINGS方法返回 (prominence, relation, weights)，权重在第3个位置
                    if len(result) >= 3:
                        return result[2]
                    else:
                        raise ValueError(f"Unexpected result format for {method}")
                elif method == 'ahp':
                    # AHP方法返回 (weights, consistency_ratio)
                    return result[0]
                elif method == 'simplified_bwm':
                    # 简化BWM方法返回 (CR, weights)
                    return result[1] if len(result) > 1 else result[0]
                else:
                    # 部分方法（如 AHP）返回 (weights, consistency_ratio)
                    return result[0]
            elif isinstance(result, np.ndarray):
                return result
            else:
                return result
        else:
            # 返回所有方法的权重
            return self.result_manager.get_all_weights()
    
    def get_consistency_ratios(self) -> Dict[str, float]:
        """
        获取提供一致性指标的方法的 CR 值。
        
        返回:
            方法名称到一致性比率的映射字典
        """
        ratios = {}
        
        # AHP 提供一致性比率 CR
        if 'ahp' in self.results:
            result = self.results['ahp']
            if isinstance(result, tuple) and len(result) > 1:
                ratios['ahp'] = result[1]
        
        # 简化 BWM 提供 CR（返回元组第一个元素）
        if 'simplified_bwm' in self.results:
            result = self.results['simplified_bwm']
            if isinstance(result, tuple) and len(result) > 1:
                ratios['simplified_bwm'] = result[0]  # CR is first element
        
        return ratios
    
    def compare_weights(self, methods: Optional[List[str]] = None) -> Any:
        """
        跨方法比较权重（返回 DataFrame）。
        """
        return self.result_manager.compare_weights(methods)
    
