"""
Result management for decision analysis methods.

Provides unified storage, processing and access for outputs such as weights,
scores, rankings and matrices, with basic export utilities.
"""

import numpy as np
import json
import pandas as pd
import os
import warnings
from datetime import datetime
from scipy.stats import kendalltau
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass, field


class ResultType(Enum):
    """Enumeration of possible result types from decision methods."""
    WEIGHTS = "weights"
    SCORES = "scores"
    RANKINGS = "rankings"
    MATRIX = "matrix"
    CLASSIFICATION = "classification"
    COMPREHENSIVE = "comprehensive"


@dataclass
class MethodResult:
    """
    Container for a single method result.

    Attributes:
        - method_name: Name of the method.
        - raw_output: Raw output from the method.
        - processed_data: Structured representation (vector/matrix/tuple, etc.).
        - metadata: Additional information (timestamp, parameters, etc.).
    """
    method_name: str
    raw_output: Any
    processed_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-init: convert raw output into structured form."""
        self._process_output()
    
    def _process_output(self):
        """Process raw output based on method name and expected formats."""
        method_name = self.method_name.lower()
        
        # 权重类方法处理
        if self._is_weighting_method(method_name):
            self._process_weighting_output(method_name)
        # 模糊权重类方法处理
        elif self._is_fuzzy_weighting_method(method_name):
            self._process_fuzzy_weighting_output(method_name)
        # 评分类方法处理
        elif self._is_scoring_method(method_name):
            self._process_scoring_output(method_name)
        # 模糊评分类方法处理
        elif self._is_fuzzy_scoring_method(method_name):
            self._process_fuzzy_scoring_output(method_name)
        # 成对比较类方法处理
        elif self._is_pairwise_method(method_name):
            self._process_pairwise_output(method_name)
        # 分类方法处理
        elif self._is_classification_method(method_name):
            self._process_classification_output(method_name)
        else:
            # 默认处理
            self._process_default_output()
    
    def get_weights(self) -> Optional[np.ndarray]:
        """Return weights if available."""
        if 'weights' in self.processed_data:
            return self.processed_data['weights']
        return None
    
    def get_scores(self) -> Optional[np.ndarray]:
        """Return scores if available."""
        if 'scores' in self.processed_data:
            return self.processed_data['scores']
        return None
    
    def get_ranking(self) -> Optional[np.ndarray]:
        """Return or compute ranking if available."""
        if 'ranking' in self.processed_data:
            return self.processed_data['ranking']
        
        # 尝试根据评分计算排名
        scores = self.get_scores()
        if scores is not None:
            # 分数越高，名次越靠前（1 为最佳）
            return np.argsort(-scores) + 1
        
        # 尝试根据权重计算排名
        weights = self.get_weights()
        if weights is not None:
            return np.argsort(-weights) + 1
        
        return None
    
    def _is_weighting_method(self, method_name: str) -> bool:
        """判断是否为权重类方法。"""
        weighting_methods = {
            'ahp', 'bwm', 'simplified_bwm', 'fucom', 'roc', 'rrw', 'rsw',
            'critic', 'entropy', 'idocriw', 'merec', 'mpsi', 'seca', 'cilos',
            'dematel', 'wings'
        }
        return method_name in weighting_methods
    
    def _is_fuzzy_weighting_method(self, method_name: str) -> bool:
        """判断是否为模糊权重类方法。"""
        fuzzy_weighting_methods = {
            'fuzzy_ahp', 'fuzzy_bwm', 'fuzzy_fucom', 'fuzzy_critic', 'fuzzy_merec'
        }
        return method_name in fuzzy_weighting_methods
    
    def _is_scoring_method(self, method_name: str) -> bool:
        """判断是否为评分类方法。"""
        scoring_methods = {
            'topsis', 'vikor', 'saw', 'aras', 'copras', 'edas', 'codas', 'cocoso',
            'cradis', 'mabac', 'mairca', 'marcos', 'moora', 'moosra', 'multimoora',
            'waspas', 'wisp', 'todim', 'gra', 'lmaw', 'rafsi', 'spotis', 'macbeth',
            'mara', 'ocra', 'oreste', 'piv', 'regime', 'rov', 'borda', 'copeland',
            'smart', 'maut', 'psi'
        }
        return method_name in scoring_methods
    
    def _is_fuzzy_scoring_method(self, method_name: str) -> bool:
        """判断是否为模糊评分类方法。"""
        fuzzy_scoring_methods = {
            'fuzzy_aras', 'fuzzy_copras', 'fuzzy_edas', 'fuzzy_moora',
            'fuzzy_ocra', 'fuzzy_topsis', 'fuzzy_vikor', 'fuzzy_waspas'
        }
        return method_name in fuzzy_scoring_methods
    
    def _is_pairwise_method(self, method_name: str) -> bool:
        """判断是否为成对比较类方法。"""
        pairwise_methods = {
            'electre_iii', 'electre_iv', 'promethee_ii'
        }
        return method_name in pairwise_methods

    def _is_classification_method(self, method_name: str) -> bool:
        """判断是否为分类类方法。"""
        classification_methods = {
            'electre_tri_b', 'cpp_tri'
        }
        return method_name in classification_methods
    
    def _process_weighting_output(self, method_name: str):
        """处理权重类方法的输出。"""
        if method_name in ['ahp', 'simplified_bwm']:
            # 这些方法返回元组：(权重, 一致性分数)
            if isinstance(self.raw_output, tuple) and len(self.raw_output) >= 2:
                self.processed_data['weights'] = self.raw_output[0]
                self.processed_data['consistency_ratio'] = self.raw_output[1]
                self.processed_data['type'] = 'weighting_with_consistency'
            else:
                # 如果格式不符合预期，按普通权重处理
                self.processed_data['weights'] = self.raw_output
                self.processed_data['type'] = 'weighting'
        elif method_name in ['dematel', 'wings']:
            # DEMATEL和WINGS方法返回元组：(关系矩阵1, 关系矩阵2, 权重)，权重在第3个位置
            if isinstance(self.raw_output, tuple) and len(self.raw_output) >= 3:
                self.processed_data['weights'] = self.raw_output[2]
                if method_name == 'dematel':
                    self.processed_data['prominence'] = self.raw_output[0]  # D_plus_R
                    self.processed_data['relation'] = self.raw_output[1]    # D_minus_R
                    self.processed_data['type'] = 'dematel_weighting'
                elif method_name == 'wings':
                    self.processed_data['prominence'] = self.raw_output[0]  # r_plus_c
                    self.processed_data['relation'] = self.raw_output[1]    # r_minus_c
                    self.processed_data['type'] = 'wings_weighting'
            else:
                # 如果格式不符合预期，按普通权重处理
                self.processed_data['weights'] = self.raw_output
                self.processed_data['type'] = 'weighting'
        else:
            # 其他权重方法直接返回权重数组
            self.processed_data['weights'] = self.raw_output
            self.processed_data['type'] = 'weighting'
    
    def _process_fuzzy_weighting_output(self, method_name: str):
        """处理模糊权重类方法的输出。"""
        if method_name == 'fuzzy_ahp':
            # fuzzy_ahp: (f_w, d_w, n_w, rc)
            if isinstance(self.raw_output, tuple) and len(self.raw_output) >= 4:
                self.processed_data['fuzzy_weights'] = self.raw_output[0]
                self.processed_data['weights'] = self.raw_output[1]  # 解模糊权重
                self.processed_data['normalized_weights'] = self.raw_output[2]
                self.processed_data['consistency_ratio'] = self.raw_output[3]
                self.processed_data['type'] = 'fuzzy_weighting'
            else:
                self._process_default_output()
        elif method_name == 'fuzzy_bwm':
            # fuzzy_bwm: (f_w, d_w, e_v, rc)
            if isinstance(self.raw_output, tuple) and len(self.raw_output) >= 4:
                self.processed_data['fuzzy_weights'] = self.raw_output[0]
                self.processed_data['weights'] = self.raw_output[1]  # 解模糊权重
                self.processed_data['consistency_indicator'] = self.raw_output[2]
                self.processed_data['consistency_ratio'] = self.raw_output[3]
                self.processed_data['type'] = 'fuzzy_weighting'
            else:
                self._process_default_output()
        elif method_name == 'fuzzy_fucom':
            # fuzzy_fucom: (f_w, d_w)
            if isinstance(self.raw_output, tuple) and len(self.raw_output) >= 2:
                self.processed_data['fuzzy_weights'] = self.raw_output[0]
                self.processed_data['weights'] = self.raw_output[1]  # 解模糊权重
                self.processed_data['type'] = 'fuzzy_weighting'
            else:
                self._process_default_output()
        else:
            # fuzzy_critic, fuzzy_merec等直接返回解模糊权重
            self.processed_data['weights'] = self.raw_output
            self.processed_data['type'] = 'fuzzy_weighting'
    
    def _process_scoring_output(self, method_name: str):
        """处理评分类方法的输出。"""
        if method_name == 'vikor':
            # vikor: (flow_s, flow_r, flow_q, solution) - flow_q是最终结果
            if isinstance(self.raw_output, tuple) and len(self.raw_output) >= 3:
                flow_q = self.raw_output[2]
                self._extract_scores_and_ranking(flow_q)
                self.processed_data['flow_s'] = self.raw_output[0]
                self.processed_data['flow_r'] = self.raw_output[1]
                self.processed_data['flow_q'] = flow_q
                if len(self.raw_output) >= 4:
                    self.processed_data['solution'] = self.raw_output[3]
                self.processed_data['type'] = 'scoring_vikor'
            else:
                self._process_default_output()
        elif method_name == 'multimoora':
            # multimoora: (flow_1, flow_2, flow_3, flow_final) - flow_final是最终结果
            if isinstance(self.raw_output, tuple) and len(self.raw_output) >= 4:
                flow_final = self.raw_output[3]
                self._extract_scores_and_ranking(flow_final)
                self.processed_data['flow_1'] = self.raw_output[0]
                self.processed_data['flow_2'] = self.raw_output[1]
                self.processed_data['flow_3'] = self.raw_output[2]
                self.processed_data['flow_final'] = flow_final
                self.processed_data['type'] = 'scoring_multimoora'
            else:
                self._process_default_output()
        elif method_name == 'waspas':
            # waspas: (wsm, wpm, waspas) - waspas是最终结果
            if isinstance(self.raw_output, tuple) and len(self.raw_output) >= 3:
                waspas_result = self.raw_output[2]
                self._extract_scores_and_ranking(waspas_result)
                self.processed_data['wsm'] = self.raw_output[0]
                self.processed_data['wpm'] = self.raw_output[1]
                self.processed_data['waspas'] = waspas_result
                self.processed_data['type'] = 'scoring_waspas'
            else:
                self._process_default_output()
        else:
            # 通用评分方法：二维数组，第一列方案编号，第二列得分
            self._extract_scores_and_ranking(self.raw_output)
            self.processed_data['type'] = 'scoring_standard'
    
    def _process_fuzzy_scoring_output(self, method_name: str):
        """处理模糊评分类方法的输出。"""
        if method_name == 'fuzzy_vikor':
            # fuzzy_vikor: (flow_s, flow_r, flow_q, solution) - flow_q是最终结果
            if isinstance(self.raw_output, tuple) and len(self.raw_output) >= 3:
                flow_q = self.raw_output[2]
                self._extract_scores_and_ranking(flow_q)
                self.processed_data['flow_s'] = self.raw_output[0]
                self.processed_data['flow_r'] = self.raw_output[1]
                self.processed_data['flow_q'] = flow_q
                if len(self.raw_output) >= 4:
                    self.processed_data['solution'] = self.raw_output[3]
                self.processed_data['type'] = 'fuzzy_scoring_vikor'
            else:
                self._process_default_output()
        elif method_name == 'fuzzy_waspas':
            # fuzzy_waspas: (wsm, wpm, waspas) - waspas是最终结果
            if isinstance(self.raw_output, tuple) and len(self.raw_output) >= 3:
                waspas_result = self.raw_output[2]
                self._extract_scores_and_ranking(waspas_result)
                self.processed_data['wsm'] = self.raw_output[0]
                self.processed_data['wpm'] = self.raw_output[1]
                self.processed_data['waspas'] = waspas_result
                self.processed_data['type'] = 'fuzzy_scoring_waspas'
            else:
                self._process_default_output()
        else:
            # 其他模糊评分方法：二维数组，第一列方案编号，第二列得分
            self._extract_scores_and_ranking(self.raw_output)
            self.processed_data['type'] = 'fuzzy_scoring_standard'
    
    def _process_pairwise_output(self, method_name: str):
        """处理成对比较类方法的输出。"""
        if method_name == 'electre_iii':
            # electre_iii: 6个参数，rank_M（索引4）是最终排序
            if isinstance(self.raw_output, tuple) and len(self.raw_output) >= 5:
                rank_m = self.raw_output[4]
                self._extract_ranking_from_list(rank_m)
                self.processed_data['global_concordance'] = self.raw_output[0]
                self.processed_data['credibility'] = self.raw_output[1]
                self.processed_data['rank_d'] = self.raw_output[2]
                self.processed_data['rank_a'] = self.raw_output[3]
                self.processed_data['rank_m'] = rank_m
                if len(self.raw_output) >= 6:
                    self.processed_data['rank_p'] = self.raw_output[5]
                self.processed_data['type'] = 'pairwise_electre_iii'
            else:
                self._process_default_output()
        elif method_name == 'electre_iv':
            # electre_iv: 5个参数，rank_M（索引3）是最终排序
            if isinstance(self.raw_output, tuple) and len(self.raw_output) >= 4:
                rank_m = self.raw_output[3]
                self._extract_ranking_from_list(rank_m)
                self.processed_data['credibility'] = self.raw_output[0]
                self.processed_data['rank_d'] = self.raw_output[1]
                self.processed_data['rank_a'] = self.raw_output[2]
                self.processed_data['rank_m'] = rank_m
                if len(self.raw_output) >= 5:
                    self.processed_data['rank_p'] = self.raw_output[4]
                self.processed_data['type'] = 'pairwise_electre_iv'
            else:
                self._process_default_output()
        elif method_name == 'promethee_ii':
            # promethee_ii: flow数组，第一列排序，第二列净流量值
            self._extract_scores_and_ranking(self.raw_output)
            self.processed_data['flow'] = self.raw_output
            self.processed_data['type'] = 'pairwise_promethee_ii'
        else:
            self._process_default_output()
    
    def _process_classification_output(self, method_name: str):
        """处理分类方法的输出。"""
        if method_name == 'electre_tri_b':
            self.processed_data['classification'] = self.raw_output
            self.processed_data['type'] = 'classification_electre_tri_b'
        elif method_name == 'cpp_tri':
            self.processed_data['classification'] = self.raw_output
            self.processed_data['type'] = 'classification_cpp_tri'
        else:
            self._process_default_output()
    
    def _extract_scores_and_ranking(self, data):
        """从标准格式的评分数据中提取分数和排名。"""
        if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 2:
            # 标准格式：第一列是方案编号，第二列是得分
            scores = data[:, 1]
            # 根据得分计算排名（得分越高排名越靠前）
            ranking = np.argsort(-scores) + 1
            
            self.processed_data['scores'] = scores
            self.processed_data['ranking'] = ranking
            self.processed_data['raw_scores_data'] = data
        else:
            # 非标准格式，尝试其他解析方式
            if isinstance(data, (list, np.ndarray)):
                self.processed_data['raw_data'] = data
    
    def _extract_ranking_from_list(self, rank_list):
        """从排序列表中提取排名信息。"""
        if isinstance(rank_list, list):
            # 将列表形式的排序转换为数值排名
            try:
                # 处理可能的并列情况（如'a1; a2'）
                ranking_data = []
                for i, item in enumerate(rank_list):
                    if isinstance(item, str):
                        # 提取方案编号
                        if ';' in item:
                            # 并列情况
                            alternatives = [alt.strip() for alt in item.split(';')]
                            for alt in alternatives:
                                alt_num = int(alt.replace('a', '')) if 'a' in alt else i + 1
                                ranking_data.append((alt_num, i + 1))
                        else:
                            alt_num = int(item.replace('a', '')) if 'a' in item else i + 1
                            ranking_data.append((alt_num, i + 1))
                
                # 根据方案编号排序，提取排名
                ranking_data.sort(key=lambda x: x[0])
                ranking = np.array([rank for _, rank in ranking_data])
                self.processed_data['ranking'] = ranking
            except (ValueError, TypeError):
                # 如果解析失败，保存原始数据
                self.processed_data['raw_ranking'] = rank_list
        else:
            self.processed_data['raw_ranking'] = rank_list
    
    def _process_default_output(self):
        """默认处理方式。"""
        if isinstance(self.raw_output, np.ndarray):
            if self.raw_output.ndim == 1:
                self.processed_data['values'] = self.raw_output
                self.processed_data['type'] = 'vector'
            elif self.raw_output.ndim == 2:
                self.processed_data['values'] = self.raw_output
                self.processed_data['type'] = 'matrix'
        elif isinstance(self.raw_output, tuple):
            self.processed_data['values'] = self.raw_output
            self.processed_data['type'] = 'tuple'
        else:
            self.processed_data['values'] = self.raw_output
            self.processed_data['type'] = 'other'

class ResultManager:
    """
    管理多个决策方法的结果。
    
    提供能力：
    - 多方法结果的统一存储
    - 常见输出（权重、评分、排名）的统一访问接口
    - 结果的比较、聚合与一致性分析
    - 多格式导出能力
    """
    
    def __init__(self):
        """初始化结果管理器。"""
        self.results: Dict[str, MethodResult] = {}
        self.metadata: Dict[str, Any] = {}
    
    def add_result(self, method_name: str, raw_output: Any, 
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        添加某个方法执行后的结果。
        
        参数：
            - method_name: 方法名称
            - raw_output: 方法的原始输出
            - metadata: 关于该结果的可选元信息
        """
        result = MethodResult(
            method_name=method_name,
            raw_output=raw_output,
            metadata=metadata or {}
        )
        self.results[method_name] = result
    
    def get_result(self, method_name: str) -> Optional[MethodResult]:
        """
        获取指定方法的结果。
        
        参数：
            - method_name: 方法名称
        返回：
            - 对应的 `MethodResult` 对象；若不存在则返回 None
        """
        return self.results.get(method_name)
    
    def store_weighting_results(self, results: Dict[str, Any]) -> None:
        """
        批量存储权重计算结果。
        
        Args:
            results: 权重结果字典，格式为 {method_name: raw_output, ...}
        """
        for method_name, raw_output in results.items():
            self.add_result(method_name, raw_output)
    
    def store_scoring_results(self, results: Dict[str, Any]) -> None:
        """
        批量存储评分结果。
        
        Args:
            results: 评分结果字典，格式为 {method_name: raw_output, ...}
        """
        for method_name, raw_output in results.items():
            self.add_result(method_name, raw_output)
    
    def store_pairwise_results(self, results: Dict[str, Any]) -> None:
        """批量存储成对比较结果。"""
        for method_name, raw_output in results.items():
            self.add_result(method_name, raw_output)
    
    def store_fuzzy_results(self, results: Dict[str, Any]) -> None:
        """批量存储模糊方法结果。"""
        for method_name, raw_output in results.items():
            self.add_result(method_name, raw_output)
    
    def get_all_results(self) -> Dict[str, Any]:
        """
        获取所有结果的统一接口。
        
        Returns:
            包含所有结果的字典
        """
        return {name: result.processed_data for name, result in self.results.items()}
    
    def get_all_weights(self) -> Dict[str, np.ndarray]:
        """
        获取所有提供了权重输出的方法的权重。
        
        返回：
            - 字典，键为方法名，值为对应的权重向量
        """
        weights = {}
        for name, result in self.results.items():
            w = result.get_weights()
            if w is not None:
                weights[name] = w
        return weights
    
    def get_weights(self, method_name: Optional[str] = None) -> Union[Dict[str, np.ndarray], np.ndarray, None]:
        """
        获取权重结果。
        
        Args:
            method_name: 方法名称，如果为None则返回所有方法的权重
            
        Returns:
            如果指定了method_name，返回该方法的权重数组；
            如果未指定，返回所有方法权重的字典
        """
        if method_name is None:
            return self.get_all_weights()
        else:
            if method_name in self.results:
                return self.results[method_name].get_weights()
            return None
    
    def get_all_scores(self) -> Dict[str, np.ndarray]:
        """
        获取所有提供了评分输出的方法的评分。
        
        返回：
            - 字典，键为方法名，值为对应的评分向量
        """
        scores = {}
        for name, result in self.results.items():
            s = result.get_scores()
            if s is not None:
                scores[name] = s
        return scores
    
    def get_all_rankings(self) -> Dict[str, np.ndarray]:
        """
        获取所有方法的排名。
        
        返回：
            - 字典，键为方法名，值为对应的排名向量
        """
        rankings = {}
        for name, result in self.results.items():
            r = result.get_ranking()
            if r is not None:
                rankings[name] = r
        return rankings
    
    def get_rankings(self, method_name: Optional[str] = None) -> Union[Dict[str, np.ndarray], np.ndarray, None]:
        """
        获取排名结果。
        
        Args:
            method_name: 方法名称，如果为None则返回所有方法的排名
            
        Returns:
            如果指定了method_name，返回该方法的排名数组；
            如果未指定，返回所有方法排名的字典
        """
        if method_name is None:
            return self.get_all_rankings()
        else:
            if method_name in self.results:
                return self.results[method_name].get_ranking()
            return None
    
    def compare_weights(self, methods: Optional[List[str]] = None, add_stats: bool = False) -> pd.DataFrame:
        """
        跨方法比较权重（针对权重类和模糊权重类方法）。
        
        参数：
            - methods: 需要比较的方法列表（None 表示全部权重类方法）
            - add_stats: 是否在结果中附加统计列（均值、标准差、最小值、最大值）
        返回：
            - 包含比较结果的 DataFrame，列为方法名，行为准则名
        """
        # 获取所有权重数据
        all_weights = self.get_all_weights()
        
        # 根据methods参数筛选方法
        if methods:
            weights = {method: all_weights[method] for method in methods if method in all_weights}
        else:
            weights = all_weights
        
        if not weights:
            return pd.DataFrame()
        
        # 构造 DataFrame，首先验证和处理权重数据
        df_data = {}
        valid_weights = {}
        
        # 验证每个权重数据的有效性
        for method, w in weights.items():
            if hasattr(w, '__len__') and not isinstance(w, str):
                # 是数组或列表
                valid_weights[method] = np.array(w) if not isinstance(w, np.ndarray) else w
            elif isinstance(w, (int, float, np.number)):
                # 标量值，转换为长度为1的数组（可能的单准则情况）
                valid_weights[method] = np.array([w])
            else:
                # 尝试转换为数组
                try:
                    valid_weights[method] = np.array(w)
                except:
                    warnings.warn(f"跳过方法 {method}：无法将权重数据转换为数组格式")
                    continue
        
        if not valid_weights:
            return pd.DataFrame()
        
        max_len = max(len(w) for w in valid_weights.values())
        
        for method, w in valid_weights.items():
            # 若长度不一致则进行填充对齐
            if len(w) < max_len:
                padded = np.pad(w, (0, max_len - len(w)), constant_values=np.nan)
                df_data[method] = padded
            else:
                df_data[method] = w
        
        df = pd.DataFrame(df_data)
        df.index = [f'Criterion_{i+1}' for i in range(max_len)]
        
        # 仅在需要时添加统计列
        if add_stats:
            df['Mean'] = df.mean(axis=1)
            df['Std'] = df.std(axis=1)
            df['Min'] = df.min(axis=1)
            df['Max'] = df.max(axis=1)
        
        return df
    
    def compare_scores(self, methods: Optional[List[str]] = None) -> pd.DataFrame:
        """
        跨方法比较评分（针对评分类和模糊评分类方法）。
        
        参数：
            - methods: 需要比较的方法列表（None 表示全部评分类方法）
        返回：
            - 包含比较结果的 DataFrame，列为方法名，行为方案名
        """
        # 获取所有评分数据
        all_scores = self.get_all_scores()
        
        # 根据methods参数筛选方法
        if methods:
            scores = {method: all_scores[method] for method in methods if method in all_scores}
        else:
            scores = all_scores
        
        if not scores:
            return pd.DataFrame()
        
        # 构造 DataFrame
        df_data = {}
        max_len = max(len(s) for s in scores.values())
        
        # 添加评分列
        for method, s in scores.items():
            if len(s) < max_len:
                padded = np.pad(s, (0, max_len - len(s)), constant_values=np.nan)
                df_data[method] = padded
            else:
                df_data[method] = s
        
        df = pd.DataFrame(df_data)
        df.index = [f'Alternative_{i+1}' for i in range(max_len)]
        
        return df
    
    def compare_rankings(self, methods: Optional[List[str]] = None) -> pd.DataFrame:
        """
        跨方法比较排名（针对成对比较类和其他有排名的方法）。
        
        参数：
            - methods: 需要比较的方法列表（None 表示全部有排名的方法）
        返回：
            - 包含比较结果的 DataFrame，列为方法名，行为方案名
        """
        # 获取所有排名数据
        all_rankings = self.get_all_rankings()
        excluded_methods = []
        
        # 排除分类方法
        rankings = {}
        for name, ranking in all_rankings.items():
            if name.lower() == 'electre_tri_b':
                excluded_methods.append(name)
                continue
            rankings[name] = ranking
        
        # 给出警告（如果有被排除的方法）
        if excluded_methods:
            warnings.warn(
                f"以下方法被排除在排名比较之外，因为它们是分类方法而不是排序方法: {', '.join(excluded_methods)}。",
                UserWarning
            )
        
        # 根据methods参数筛选方法
        if methods:
            rankings = {method: rankings[method] for method in methods if method in rankings}
        
        if not rankings:
            return pd.DataFrame()
        
        # 构造 DataFrame
        df_data = {}
        max_len = max(len(r) for r in rankings.values())
        
        for method, r in rankings.items():
            if len(r) < max_len:
                # 对于排名数据，使用-1作为填充值而不是NaN，以避免类型冲突
                padded = np.pad(r, (0, max_len - len(r)), constant_values=-1)
                df_data[method] = padded
            else:
                df_data[method] = r
        
        df = pd.DataFrame(df_data)
        df.index = [f'Alternative_{i+1}' for i in range(max_len)]
        
        # 将-1替换为NaN以表示缺失值，并转换为浮点数类型
        df = df.replace(-1, np.nan).astype(float)
        
        # 添加类似 Borda 的共识排名（如果有多个方法）
        if len(df.columns) > 1:
            df['Consensus'] = df.mean(axis=1, skipna=True).rank(method='min').astype(int)
        
        return df
    
    def compare_classifications(self, methods: Optional[List[str]] = None) -> pd.DataFrame:
        """
        跨方法比较分类结果（专用于方案分类，如 ELECTRE TRI-B、CPP Tri 等）。
        
        参数：
            - methods: 需要比较的方法列表（None 表示自动从已存结果中挑选分类方法）
        返回：
            - DataFrame：列为方法名，行为方案名，值为类别编号；若有多个方法，附加 Consensus 列表示多数表决结果
        """
        # 识别并收集分类结果
        classification_results: Dict[str, np.ndarray] = {}
        known_classification_methods = {"electre_tri_b", "cpp_tri"}
        for name, result in self.results.items():
            method_lower = name.lower()
            data = result.processed_data
            class_vector = None
            # 优先读取标准键
            if isinstance(data, dict) and 'classification' in data:
                class_vector = data['classification']
            # 兼容默认处理（如 cpp_tri 返回的一维向量保存在 values）
            elif method_lower in known_classification_methods and isinstance(data, dict) and 'values' in data:
                class_vector = data['values']
            # 兜底：若方法名在已知集合中且 raw_output 为一维数组/列表
            elif method_lower in known_classification_methods:
                raw_output = result.raw_output
                if isinstance(raw_output, (list, np.ndarray)):
                    class_vector = raw_output
            
            if class_vector is None:
                continue
            # 规范为一维 numpy 数组
            class_array = np.array(class_vector)
            if class_array.ndim != 1:
                continue
            classification_results[name] = class_array
        
        # 根据 methods 参数筛选
        if methods:
            classification_results = {m: classification_results[m] for m in methods if m in classification_results}
        
        if not classification_results:
            return pd.DataFrame()
        
        # 对齐不同长度（以 NaN 填充）
        max_len = max(len(v) for v in classification_results.values())
        df_data: Dict[str, np.ndarray] = {}
        for method, arr in classification_results.items():
            arr = arr.astype(float) if not np.issubdtype(arr.dtype, np.floating) else arr
            if len(arr) < max_len:
                padded = np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
                df_data[method] = padded
            else:
                df_data[method] = arr
        
        df = pd.DataFrame(df_data)
        df.index = [f'Alternative_{i+1}' for i in range(max_len)]
        
        # 多数表决的共识类别（存在多个方法时）
        if len(df.columns) > 1:
            modes = df.mode(axis=1, dropna=True)
            if not modes.empty:
                df['Consensus'] = modes.iloc[:, 0].astype(int)
        
        return df
    
    def __repr__(self) -> str:
        """对象的字符串表示。"""
        return f"ResultManager(methods={len(self.results)})"
      
    def __str__(self) -> str:
        """可读性良好的字符串表示。"""
        if not self.results:
            return "ResultManager: No results stored"
        
        methods = ', '.join(list(self.results.keys())[:5])
        if len(self.results) > 5:
            methods += f", ... ({len(self.results)} total)"
        
        return f"ResultManager: {methods}"
