"""
决策方法的参数验证器。

本模块为决策分析方法中使用的各种类型的输入参数提供全面的验证。
"""

import numpy as np
from typing import List, Union, Optional, Tuple, Any, Dict
import warnings


class ParameterValidator:
    """
    为决策分析中的通用参数提供验证方法。
    """
    
    @staticmethod
    def validate_decision_matrix(matrix: Any, min_alternatives: int = 2, 
                                min_criteria: int = 2) -> Dict[str, Any]:
        """
        验证决策矩阵。
        
        参数:
            matrix: 要验证的决策矩阵
            min_alternatives: 所需的最小方案数
            min_criteria: 所需的最小准则数
            
        返回:
            包含验证结果的字典，其中包含 'is_valid'、'processed_data'、'error_message'
            
        """
        if matrix is None:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': 'Decision matrix cannot be None'
            }
        
        # 如果需要，转换为 numpy 数组
        if not isinstance(matrix, np.ndarray):
            try:
                matrix = np.array(matrix)
            except Exception as e:
                return {
                    'is_valid': False,
                    'processed_data': None,
                    'error_message': f'Cannot convert decision matrix to numpy array: {e}'
                }
        
        # 检查维度
        if matrix.ndim != 2:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': f'决策矩阵必须是二维数组，当前维度为{matrix.ndim}'
            }
        
        n_alternatives, n_criteria = matrix.shape
        
        if n_alternatives < min_alternatives:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': f'决策矩阵至少需要{min_alternatives}个方案，当前只有{n_alternatives}个'
            }
        
        if n_criteria < min_criteria:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': f'决策矩阵至少需要{min_criteria}个准则，当前只有{n_criteria}个'
            }
        
        # 检查无效值
        if np.any(np.isnan(matrix)):
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': '决策矩阵包含缺失值（NaN）'
            }
        
        if np.any(np.isinf(matrix)):
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': '决策矩阵包含无穷值'
            }
        
        return {
            'is_valid': True,
            'processed_data': matrix,
            'error_message': None
        }
    
    @staticmethod
    def validate_weights(weights: Any, n_criteria: Optional[int] = None,
                        normalize: bool = True) -> Dict[str, Any]:
        """
        验证权重向量。
        
        参数:
            weights: 要验证的权重向量
            n_criteria: 预期的准则数（可选）
            normalize: 是否将权重归一化以使总和为 1
            
        返回:
            包含验证结果的字典，其中包含 'is_valid'、'processed_data'、'error_message'
        """
        if weights is None:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': 'Weights cannot be None'
            }
        
        # 如果需要，转换为 numpy 数组
        if not isinstance(weights, np.ndarray):
            try:
                weights = np.array(weights, dtype=float)
            except Exception as e:
                return {
                    'is_valid': False,
                    'processed_data': None,
                    'error_message': f'Cannot convert weights to numpy array: {e}'
                }
        
        # 检查维度
        if weights.ndim != 1:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': f'权重必须是一维数组，当前维度为{weights.ndim}'
            }
        
        # 如果指定，检查长度
        if n_criteria is not None and len(weights) != n_criteria:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': f'权重向量长度（{len(weights)}）与准则数量（{n_criteria}）不匹配'
            }
        
        # 检查无效值
        if np.any(np.isnan(weights)):
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': '权重包含NaN值'
            }
        
        if np.any(np.isinf(weights)):
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': '权重包含无穷值'
            }
        
        if np.any(weights < 0):
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': '权重必须是非负值'
            }
        
        if np.all(weights == 0):
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': '至少一个权重必须非零'
            }
        
        # 如果请求，进行归一化
        if normalize:
            weight_sum = np.sum(weights)
            if not np.isclose(weight_sum, 1.0, rtol=1e-3):
                weights = weights / weight_sum
                warnings.warn(
                    f"Weights normalized from sum {weight_sum:.4f} to 1.0"
                )
        
        return {
            'is_valid': True,
            'processed_data': weights,
            'error_message': None
        }
    
    @staticmethod
    def validate_criterion_type(criterion_type: Any, n_criteria: Optional[int] = None) -> Dict[str, Any]:
        """
        验证准则类型列表。
        
        参数:
            criterion_type: 准则类型列表（'max' 或 'min'）
            n_criteria: 预期的准则数（可选）
            
        返回:
            包含验证结果的字典，其中包含 'is_valid'、'processed_data'、'error_message'
        """
        if criterion_type is None:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': 'Criterion type cannot be None'
            }
        
        if not isinstance(criterion_type, list):
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': f'准则类型必须是列表，当前类型为{type(criterion_type)}'
            }
        
        # 检查空列表
        if len(criterion_type) == 0:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': '准则类型列表不能为空'
            }
        
        # 如果指定，检查长度
        if n_criteria is not None and len(criterion_type) != n_criteria:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': f'准则类型列表长度（{len(criterion_type)}）与准则数量（{n_criteria}）不匹配'
            }
        
        # 验证每个类型
        valid_types = {'max', 'min'}
        for i, ctype in enumerate(criterion_type):
            if ctype not in valid_types:
                return {
                    'is_valid': False,
                    'processed_data': None,
                    'error_message': f'索引{i}处的准则类型"{ctype}"无效，必须是max或min'
                }
        
        return {
            'is_valid': True,
            'processed_data': criterion_type,
            'error_message': None
        }
    
    @staticmethod
    def validate_dimensions_consistency(decision_matrix: np.ndarray, weights: Optional[np.ndarray] = None, 
                                      criterion_type: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        验证矩阵维度与相关参数之间的一致性。
        
        参数:
            decision_matrix: 决策矩阵
            weights: 权重向量（可选）
            criterion_type: 准则类型列表（可选）
            
        返回:
            包含验证结果的字典，其中包含 'is_valid'、'processed_data'、'error_message'
        """
        if decision_matrix is None:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': 'Decision matrix cannot be None'
            }
        
        if decision_matrix.ndim != 2:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': '决策矩阵必须是二维数组'
            }
        
        n_alternatives, n_criteria = decision_matrix.shape
        
        # 检查权重一致性
        if weights is not None:
            if len(weights) != n_criteria:
                return {
                    'is_valid': False,
                    'processed_data': None,
                    'error_message': f'维度不匹配：权重长度（{len(weights)}）与准则数量（{n_criteria}）不一致'
                }
        
        # 检查准则类型一致性
        if criterion_type is not None:
            if len(criterion_type) != n_criteria:
                return {
                    'is_valid': False,
                    'processed_data': None,
                    'error_message': f'维度不匹配：准则类型长度（{len(criterion_type)}）与准则数量（{n_criteria}）不一致'
                }
        
        return {
            'is_valid': True,
            'processed_data': {
                'n_alternatives': n_alternatives,
                'n_criteria': n_criteria
            },
            'error_message': None
        }
    
    @staticmethod
    def validate_fuzzy_number(fuzzy_num: Any, triangular: bool = True) -> Dict[str, Any]:
        """
        验证模糊数。
        
        参数:
            fuzzy_num: 要验证的模糊数
            triangular: 是否期望三角模糊数（3个值）
            
        返回:
            包含验证结果的字典，其中包含 'is_valid'、'processed_data'、'error_message'
        """
        if fuzzy_num is None:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': 'Fuzzy number cannot be None'
            }
        
        # 如果需要，转换为列表/数组
        try:
            if isinstance(fuzzy_num, (list, tuple)):
                fuzzy_array = np.array(fuzzy_num, dtype=float)
            elif isinstance(fuzzy_num, np.ndarray):
                fuzzy_array = fuzzy_num.astype(float)
            else:
                return {
                    'is_valid': False,
                    'processed_data': None,
                    'error_message': f'Fuzzy number must be list, tuple, or array, got {type(fuzzy_num)}'
                }
        except Exception as e:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': f'Cannot convert fuzzy number to numeric array: {e}'
            }
        
        # 检查三角模糊数
        if triangular:
            if len(fuzzy_array) != 3:
                return {
                    'is_valid': False,
                    'processed_data': None,
                    'error_message': f'Triangular fuzzy number must have 3 values [l, m, u], got {len(fuzzy_array)} values'
                }
            
            l, m, u = fuzzy_array
            if not (l <= m <= u):
                return {
                    'is_valid': False,
                    'processed_data': None,
                    'error_message': f'Triangular fuzzy number must satisfy l <= m <= u, got [{l}, {m}, {u}]'
                }
        
        return {
            'is_valid': True,
            'processed_data': fuzzy_array,
            'error_message': None
        }
    
    @staticmethod
    def validate_fuzzy_matrix(matrix: Any) -> Dict[str, Any]:
        """
        验证模糊决策矩阵。
        
        参数:
            matrix: 模糊决策矩阵，其中每个元素都是一个模糊数
            
        返回:
            包含验证结果的字典，其中包含 'is_valid'、'processed_data'、'error_message'
        """
        if matrix is None:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': 'Fuzzy matrix cannot be None'
            }
        
        if not isinstance(matrix, (list, np.ndarray)):
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': f'Fuzzy matrix must be list or array, got {type(matrix)}'
            }
        
        # 为保持一致性，转换为列表的列表
        if isinstance(matrix, np.ndarray):
            matrix = matrix.tolist()
        
        # 验证每个元素
        validated_matrix = []
        for i, row in enumerate(matrix):
            if not isinstance(row, (list, np.ndarray)):
                return {
                    'is_valid': False,
                    'processed_data': None,
                    'error_message': f'Row {i} must be a list or array'
                }
            
            validated_row = []
            for j, element in enumerate(row):
                validation_result = ParameterValidator.validate_fuzzy_number(element)
                if not validation_result['is_valid']:
                    return {
                        'is_valid': False,
                        'processed_data': None,
                        'error_message': f'Invalid fuzzy number at position [{i},{j}]: {validation_result["error_message"]}'
                    }
                validated_row.append(validation_result['processed_data'].tolist())
            validated_matrix.append(validated_row)
        
        return {
            'is_valid': True,
            'processed_data': validated_matrix,
            'error_message': None
        }
    
    @staticmethod
    def validate_pairwise_matrix(matrix: Any) -> Dict[str, Any]:
        """
        验证两两比较矩阵（用于 AHP 等）。
        
        参数:
            matrix: 两两比较矩阵
            
        返回:
            包含验证结果的字典，其中包含 'is_valid'、'processed_data'、'error_message'
        """
        # 首先作为常规矩阵进行验证
        basic_validation = ParameterValidator.validate_decision_matrix(matrix, min_alternatives=2, min_criteria=2)
        if not basic_validation['is_valid']:
            return basic_validation
        
        validated_matrix = basic_validation['processed_data']
        
        # 检查是否为方阵
        n_rows, n_cols = validated_matrix.shape
        if n_rows != n_cols:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': f'Pairwise comparison matrix must be square, got shape {validated_matrix.shape}'
            }
        
        # 检查对角线元素（应为 1）
        diagonal = np.diag(validated_matrix)
        if not np.allclose(diagonal, 1.0, rtol=1e-3):
            warnings.warn("Diagonal elements of pairwise matrix should be 1.0")
        
        # 检查互反性（a_ij * a_ji 应约等于 1）
        for i in range(n_rows):
            for j in range(i+1, n_cols):
                product = validated_matrix[i, j] * validated_matrix[j, i]
                if not np.isclose(product, 1.0, rtol=1e-2):
                    warnings.warn(
                        f"Pairwise matrix may not be reciprocal: "
                        f"element[{i},{j}]={validated_matrix[i,j]:.3f} * "
                        f"element[{j},{i}]={validated_matrix[j,i]:.3f} = {product:.3f} ≠ 1.0"
                    )
        
        return {
            'is_valid': True,
            'processed_data': validated_matrix,
            'error_message': None
        }
    
    @staticmethod
    def validate_thresholds(P: Any = None, Q: Any = None, V: Any = None,
                           n_criteria: Optional[int] = None) -> Dict[str, Any]:
        """
        验证 ELECTRE/PROMETHEE 阈值向量。
        
        参数:
            P: 偏好阈值
            Q: 无差异阈值  
            V: 否决阈值
            n_criteria: 预期的准则数
            
        返回:
            包含验证结果的字典，其中包含 'is_valid'、'processed_data'、'error_message'
        """
        validated_P = None
        validated_Q = None
        validated_V = None
        
        try:
            # 验证 P (偏好)
            if P is not None:
                validated_P = np.array(P, dtype=float)
                if validated_P.ndim != 1:
                    return {
                        'is_valid': False,
                        'processed_data': None,
                        'error_message': 'Preference threshold P must be 1-dimensional'
                    }
                if n_criteria and len(validated_P) != n_criteria:
                    return {
                        'is_valid': False,
                        'processed_data': None,
                        'error_message': f'P length ({len(validated_P)}) doesn\'t match criteria count ({n_criteria})'
                    }
                if np.any(validated_P < 0):
                    return {
                        'is_valid': False,
                        'processed_data': None,
                        'error_message': 'Preference thresholds must be non-negative'
                    }
            
            # 验证 Q (无差异)
            if Q is not None:
                validated_Q = np.array(Q, dtype=float)
                if validated_Q.ndim != 1:
                    return {
                        'is_valid': False,
                        'processed_data': None,
                        'error_message': 'Indifference threshold Q must be 1-dimensional'
                    }
                if n_criteria and len(validated_Q) != n_criteria:
                    return {
                        'is_valid': False,
                        'processed_data': None,
                        'error_message': f'Q length ({len(validated_Q)}) doesn\'t match criteria count ({n_criteria})'
                    }
                if np.any(validated_Q < 0):
                    return {
                        'is_valid': False,
                        'processed_data': None,
                        'error_message': 'Indifference thresholds must be non-negative'
                    }
            
            # 验证 V (否决)
            if V is not None:
                validated_V = np.array(V, dtype=float)
                if validated_V.ndim != 1:
                    return {
                        'is_valid': False,
                        'processed_data': None,
                        'error_message': 'Veto threshold V must be 1-dimensional'
                    }
                if n_criteria and len(validated_V) != n_criteria:
                    return {
                        'is_valid': False,
                        'processed_data': None,
                        'error_message': f'V length ({len(validated_V)}) doesn\'t match criteria count ({n_criteria})'
                    }
                if np.any(validated_V < 0):
                    return {
                        'is_valid': False,
                        'processed_data': None,
                        'error_message': 'Veto thresholds must be non-negative'
                    }
        
        except Exception as e:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': f'Error processing thresholds: {e}'
            }
        
        # 如果提供了所有阈值，检查逻辑关系
        if validated_P is not None and validated_Q is not None:
            if not np.all(validated_P >= validated_Q):
                warnings.warn("Preference threshold P should be >= indifference threshold Q")
        
        if validated_V is not None and validated_P is not None:
            if not np.all(validated_V >= validated_P):
                warnings.warn("Veto threshold V should be >= preference threshold P")
        
        return {
            'is_valid': True,
            'processed_data': (validated_P, validated_Q, validated_V),
            'error_message': None
        }
    
    @staticmethod
    def validate_ranking(ranking: Any, n_items: Optional[int] = None) -> Dict[str, Any]:
        """
        验证排名列表。
        
        参数:
            ranking: 排名或项目的列表
            n_items: 预期的项目数
            
        返回:
            包含验证结果的字典，其中包含 'is_valid'、'processed_data'、'error_message'
        """
        if ranking is None:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': 'Ranking cannot be None'
            }
        
        if not isinstance(ranking, (list, np.ndarray)):
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': f'Ranking must be list or array, got {type(ranking)}'
            }
        
        # 为保持一致性，转换为列表
        if isinstance(ranking, np.ndarray):
            ranking = ranking.tolist()
        
        # 如果指定，检查长度
        if n_items is not None and len(ranking) != n_items:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': f'Ranking length ({len(ranking)}) doesn\'t match expected items ({n_items})'
            }
        
        # 检查所有项目是否唯一（对于序数排名）
        if len(set(ranking)) != len(ranking):
            warnings.warn("Ranking contains duplicate values")
        
        return {
            'is_valid': True,
            'processed_data': ranking,
            'error_message': None
        }
