"""
Parameter validators for decision methods.

Provide comprehensive validation for common input types used in decision analysis.
"""

import numpy as np
from typing import List, Union, Optional, Tuple, Any, Dict
import warnings


class ParameterValidator:
    """Validation helpers for common parameters in decision analysis."""
    
    @staticmethod
    def validate_decision_matrix(matrix: Any, min_alternatives: int = 2, 
                                min_criteria: int = 2) -> Dict[str, Any]:
        """Validate decision matrix.

        Args:
            matrix: Matrix to validate.
            min_alternatives: Minimum number of alternatives required.
            min_criteria: Minimum number of criteria required.

        Returns:
            Dict with 'is_valid', 'processed_data', 'error_message'.
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
        
        # Check dimensionality
        if matrix.ndim != 2:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': f'Decision matrix must be 2-D array, got ndim={matrix.ndim}'
            }
        
        n_alternatives, n_criteria = matrix.shape
        
        if n_alternatives < min_alternatives:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': f'Decision matrix needs at least {min_alternatives} alternatives, got {n_alternatives}'
            }
        
        if n_criteria < min_criteria:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': f'Decision matrix needs at least {min_criteria} criteria, got {n_criteria}'
            }
        
        # Check invalid values
        if np.any(np.isnan(matrix)):
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': 'Decision matrix contains NaN values'
            }
        
        if np.any(np.isinf(matrix)):
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': 'Decision matrix contains infinite values'
            }
        
        return {
            'is_valid': True,
            'processed_data': matrix,
            'error_message': None
        }
    
    @staticmethod
    def validate_weights(weights: Any, n_criteria: Optional[int] = None,
                        normalize: bool = True) -> Dict[str, Any]:
        """Validate weight vector.

        Args:
            weights: Weight vector.
            n_criteria: Expected number of criteria.
            normalize: Whether to normalize weights to sum to 1.

        Returns:
            Dict with 'is_valid', 'processed_data', 'error_message'.
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
        
        # Check dimensionality
        if weights.ndim != 1:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': f'权重必须是一维数组，当前维度为{weights.ndim}'
            }
        
        # Length check
        if n_criteria is not None and len(weights) != n_criteria:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': f'Weight length ({len(weights)}) mismatches criteria count ({n_criteria})'
            }
        
        # Invalid values
        if np.any(np.isnan(weights)):
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': 'Weights contain NaN values'
            }
        
        if np.any(np.isinf(weights)):
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': 'Weights contain infinite values'
            }
        
        if np.any(weights < 0):
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': 'Weights must be non-negative'
            }
        
        if np.all(weights == 0):
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': 'At least one weight must be non-zero'
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
        """Validate criterion type list ('max' or 'min').

        Args:
            criterion_type: List of types ('max' or 'min').
            n_criteria: Expected number of criteria.

        Returns:
            Dict with 'is_valid', 'processed_data', 'error_message'.
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
                'error_message': f'criterion_type must be a list, got {type(criterion_type)}'
            }
        
        # 检查空列表
        if len(criterion_type) == 0:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': 'criterion_type list cannot be empty'
            }
        
        # 如果指定，检查长度
        if n_criteria is not None and len(criterion_type) != n_criteria:
            return {
                'is_valid': False,
                'processed_data': None,
                'error_message': f'criterion_type length ({len(criterion_type)}) mismatches criteria count ({n_criteria})'
            }
        
        # 验证每个类型
        valid_types = {'max', 'min'}
        for i, ctype in enumerate(criterion_type):
            if ctype not in valid_types:
                return {
                    'is_valid': False,
                    'processed_data': None,
                    'error_message': f'Invalid criterion type "{ctype}" at index {i}; must be max or min'
                }
        
        return {
            'is_valid': True,
            'processed_data': criterion_type,
            'error_message': None
        }
    
    @staticmethod
    def validate_dimensions_consistency(decision_matrix: np.ndarray, weights: Optional[np.ndarray] = None, 
                                      criterion_type: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate dimensional consistency among matrix, weights and types.

        Args:
            decision_matrix: Decision matrix.
            weights: Optional weight vector.
            criterion_type: Optional list of criterion types.

        Returns:
            Dict with 'is_valid', 'processed_data', 'error_message'.
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
