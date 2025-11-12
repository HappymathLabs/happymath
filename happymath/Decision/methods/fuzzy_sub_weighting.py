"""
模糊主观权重确定方法集合。

本模块实现基于专家判断的模糊权重计算方法（如 Fuzzy AHP/BWM/FUCOM 等），
统一封装输入校验、方法调度与结果访问接口，便于在多准则场景中获取基于主观判断的模糊准则权重。
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Any, Union

from ..algorithm import fuzzy_ahp, fuzzy_bwm, fuzzy_fucom

from ..core.base import DecisionBase
from ..core.method_registry import MethodRegistry
from ..core.validators import ParameterValidator
from ..core.utils import execute_algorithm_with_suppression
from ..results.result_manager import ResultManager

class FuzzySubWeighting(DecisionBase):
    """
    模糊主观权重方法集合。
    
    基于专家判断的模糊权重确定方法，如成对比较或直接排序。
    提供统一的调用与结果比较、聚合能力。
    """
    
    # 模糊主观权重方法的注册表
    _METHOD_REGISTRY = MethodRegistry.FUZZY_SUB_WEIGHTING_METHODS
    
    # 方法到其必需参数的映射
    _METHOD_MAP = {
        'fuzzy_ahp': ['dataset'],  # 模糊成对比较矩阵
        'fuzzy_bwm': ['mic', 'lic'],  # 最佳到其他、其他到最差向量
        'fuzzy_fucom': ['criteria_rank', 'criteria_priority']  # 排序和优先级
    }
    
    _ALGORITHM_MAP = {
        'fuzzy_ahp': fuzzy_ahp,
        'fuzzy_bwm': fuzzy_bwm,
        'fuzzy_fucom': fuzzy_fucom
    }

    def __init__(self, methods: Optional[Union[str, List[str]]] = None):
        """
        初始化模糊主观权重方法集合。
        
        参数:
            methods: 指定要执行的方法（字符串或列表）。为 None 时自动选择可用方法。
        """
        super().__init__(methods)
        self.result_manager = ResultManager()
    
    def decide(self, **kwargs) -> 'FuzzySubWeighting':
        """
        按传入参数执行模糊主观权重计算方法。
        
        常用参数:
            dataset: 模糊成对比较矩阵 (用于Fuzzy AHP)
            mic: Fuzzy BWM 的最优-其他向量
            lic: Fuzzy BWM 的其他-最差向量  
            criteria_rank: 准则重要性排序 (用于Fuzzy FUCOM)
            criteria_priority: 准则优先级数值 (用于Fuzzy FUCOM)
            
        返回:
            返回 self 以便链式调用
        """
        # 校验通用参数
        kwargs_for_validation = kwargs.copy()
        if 'dataset' in kwargs_for_validation:
            del kwargs_for_validation['dataset'] # a fuzzy dataset can be 3D
        self._validate_common_parameters(**kwargs_for_validation)
        
        # 保存最近一次参数
        self._last_params = kwargs.copy()
        
        # 选择执行方法：优先严格校验用户显式指定，否则自动筛选
        if self._user_methods:
            # 用户指定了特定方法，使用严格验证
            methods_to_run = self._validate_user_methods(self._user_methods, **kwargs)
        else:
            # 自动选择方法
            methods_to_run = self._get_applicable_methods(**kwargs)
            if not methods_to_run:
                warnings.warn("No applicable fuzzy subjective weighting methods found for provided parameters")
                return self
        
        # 逐个执行并记录结果
        results_to_store = {}
        for method_name in methods_to_run:
            try:
                result = self._execute_method(method_name, **kwargs)
                self._store_result(method_name, result)
                results_to_store[method_name] = result
            except Exception as e:
                warnings.warn(f"Failed to execute {method_name}: {str(e)}")
        
        # 使用专门的模糊权重存储方法（实际调用store_fuzzy_results）
        if results_to_store:
            self.result_manager.store_fuzzy_results(results_to_store)
        
        return self
    
    def _execute_method(self, method_name: str, **kwargs) -> Any:
        """
        调度并执行具体的模糊主观权重计算方法。
        
        参数:
            method_name: 方法名称
            **kwargs: 该方法的入参
        返回:
            方法执行结果
        """
        if method_name not in self._ALGORITHM_MAP:
            raise ValueError(f"Unknown fuzzy subjective weighting method: {method_name}")

        algorithm_func = self._ALGORITHM_MAP[method_name]
        
        # 校验并准备参数
        validated_params = self._validate_fuzzy_subj_inputs(method_name, **kwargs)

        # 构造最终传递给算法的参数字典
        final_params = {'verbose': False}
        
        final_params.update(kwargs)
        final_params.update(validated_params)
        
        # 特殊参数处理
        if 'mic' in final_params and isinstance(final_params['mic'], np.ndarray):
            final_params['mic'] = final_params['mic'].tolist()
        if 'lic' in final_params and isinstance(final_params['lic'], np.ndarray):
            final_params['lic'] = final_params['lic'].tolist()

        # 执行算法
        result = execute_algorithm_with_suppression(algorithm_func, final_params)
        
        return result

    def _validate_fuzzy_subj_inputs(self, method_name: str, **kwargs) -> Dict[str, Any]:
        """校验并标准化模糊主观权重方法的输入。"""
        validated_params = {}
        
        if method_name == 'fuzzy_ahp':
            dataset = kwargs.get('dataset')
            if dataset is None:
                raise ValueError("Fuzzy AHP requires 'dataset' (fuzzy pairwise comparison matrix)")
            val_res = ParameterValidator.validate_fuzzy_matrix(dataset)
            if not val_res['is_valid']:
                raise ValueError(val_res['error_message'])
            validated_params['dataset'] = val_res['processed_data']
        
        elif method_name == 'fuzzy_bwm':
            mic = kwargs.get('mic')
            lic = kwargs.get('lic')
            if mic is None or lic is None:
                raise ValueError("Fuzzy BWM requires 'mic' and 'lic' vectors")

            def to_fuzzy_tuple_list(vec):
                # 接受 list/tuple/ndarray，标准化为 [(l,m,u), ...] 且元素为 float
                normalized = []
                for item in list(vec):
                    if isinstance(item, (list, tuple, np.ndarray)) and len(item) == 3:
                        l, m, u = item
                        normalized.append((float(l), float(m), float(u)))
                    else:
                        raise ValueError("Each element in 'mic'/'lic' must be a length-3 fuzzy tuple (l, m, u)")
                return normalized

            validated_params['mic'] = to_fuzzy_tuple_list(mic)
            validated_params['lic'] = to_fuzzy_tuple_list(lic)

        elif method_name == 'fuzzy_fucom':
            criteria_rank = kwargs.get('criteria_rank')
            criteria_priority = kwargs.get('criteria_priority')
            if criteria_rank is None or criteria_priority is None:
                raise ValueError("Fuzzy FUCOM requires 'criteria_rank' and 'criteria_priority'")

            # 确保 criteria_rank 是字符串列表
            if isinstance(criteria_rank, np.ndarray):
                processed_rank = [str(item) for item in criteria_rank]
            else:
                processed_rank = [str(item) for item in list(criteria_rank)]
            validated_params['criteria_rank'] = processed_rank

            # 确保 criteria_priority 是浮点数元组列表
            def to_fuzzy_tuple_list(vec):
                normalized = []
                for item in list(vec):
                    if isinstance(item, (list, tuple, np.ndarray)) and len(item) == 3:
                        l, m, u = item
                        normalized.append((float(l), float(m), float(u)))
                    else:
                        raise ValueError("Each element in 'criteria_priority' must be a length-3 fuzzy tuple (l, m, u)")
                return normalized
            
            processed_priority = to_fuzzy_tuple_list(criteria_priority)

            # 变通方案：填充 criteria_priority 以匹配 criteria_rank 的长度
            # 底层算法 `fuzzy_fucom` 错误地假定它们长度相同
            if len(processed_priority) < len(processed_rank):
                padding_needed = len(processed_rank) - len(processed_priority)
                processed_priority.extend([(1.0, 1.0, 1.0)] * padding_needed)

            validated_params['criteria_priority'] = processed_priority

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
            
            # Fuzzy AHP 返回 (fuzzy_weights, defuzzified_weights, normalized_weights, cr)
            if method == 'fuzzy_ahp' and isinstance(result, tuple):
                return result[0]  # Fuzzy weights
            
            # Fuzzy BWM 返回元组 (epsilon, CR, f_weights, weights)，需要获取第3个元素
            if method == 'fuzzy_bwm' and isinstance(result, tuple) and len(result) > 2:
                return result[2]  # 返回模糊权重
            
            return result
        else:
            # 返回所有模糊方法的权重
            all_weights = {}
            for name, result in self.results.items():
                if 'fuzzy' in name.lower():
                    all_weights[name] = result
            return all_weights
    
    def get_defuzzified_weights(self, method: Optional[str] = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        获取去模糊化后的（清晰的）权重。
        
        参数:
            method: 指定方法名；为 None 时返回全部方法的权重字典
            
        返回:
            去模糊化权重数组，或方法名到权重的映射字典
        """
        if method:
            result = self.results.get(method)
            if result is None:
                raise ValueError(f"No results found for method: {method}")
            
            # Fuzzy AHP 的元组结果中包含已归一化的清晰权重
            if method == 'fuzzy_ahp' and isinstance(result, tuple) and len(result) > 2:
                return result[2]  # Normalized weights
            
            # Fuzzy BWM 返回元组 (epsilon, CR, f_weights, weights)，需要获取第4个元素
            if method == 'fuzzy_bwm' and isinstance(result, tuple) and len(result) > 3:
                return result[3]  # 返回去模糊化权重
            
            return result
        else:
            # 返回所有方法的清晰权重
            return self.result_manager.get_all_weights()
    
    def get_weights(self, method: Optional[str] = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        获取各方法计算得到的权重向量（去模糊化后的权重）。
        
        参数:
            method: 指定方法名；None 时返回所有方法的权重字典
        返回:
            单个权重向量或方法名到权重的映射
        """
        return self.get_defuzzified_weights(method)
    
    def get_consistency_ratios(self) -> Dict[str, float]:
        """
        获取提供一致性指标的方法的 CR 值。
        
        返回:
            方法名称到一致性比率的映射字典
        """
        ratios = {}
        
        # Fuzzy AHP 提供一致性比率 CR
        if 'fuzzy_ahp' in self.results:
            result = self.results['fuzzy_ahp']
            if isinstance(result, tuple) and len(result) > 3:
                ratios['fuzzy_ahp'] = result[3]  # CR is last element
        
        # Fuzzy BWM 提供 CR（返回元组第二个元素）
        if 'fuzzy_bwm' in self.results:
            result = self.results['fuzzy_bwm']
            if isinstance(result, tuple) and len(result) > 1:
                ratios['fuzzy_bwm'] = result[1]  # CR is second element
        
        return ratios
    
    def compare_weights(self, methods: Optional[List[str]] = None) -> Any:
        """
        跨方法比较权重（返回 DataFrame）。
        """
        return self.result_manager.compare_weights(methods)
        
