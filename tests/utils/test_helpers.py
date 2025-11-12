"""
测试工具函数
"""
import numpy as np
import pandas as pd
import time
import traceback
import warnings
from typing import Any, Dict, List, Tuple, Optional, Union


class TestHelper:
    """测试辅助工具类"""
    
    @staticmethod
    def suppress_warnings():
        """抑制测试期间的警告"""
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
    
    @staticmethod
    def is_valid_weights(weights: np.ndarray, tolerance: float = 1e-6) -> bool:
        """
        检查权重是否有效
        
        参数:
            weights: 权重向量
            tolerance: 容差
        
        返回:
            bool: 权重是否有效
        """
        if weights is None:
            return False
        if not isinstance(weights, np.ndarray):
            return False
        if len(weights.shape) != 1:
            return False
        if np.any(weights < 0):
            return False
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            return False
        # 权重和应该约等于1（允许小的误差）
        return abs(np.sum(weights) - 1.0) < tolerance
    
    @staticmethod
    def is_valid_ranking(ranking: np.ndarray, n_alternatives: int) -> bool:
        """
        检查排名是否有效
        
        参数:
            ranking: 排名数组
            n_alternatives: 备选方案数量
        
        返回:
            bool: 排名是否有效
        """
        if ranking is None:
            return False
        if not isinstance(ranking, np.ndarray):
            return False
        if len(ranking) != n_alternatives:
            return False
        if np.any(np.isnan(ranking)) or np.any(np.isinf(ranking)):
            return False
        # 排名应该在1到n_alternatives之间
        return np.all(ranking >= 1) and np.all(ranking <= n_alternatives)
    
    @staticmethod
    def is_valid_scores(scores: np.ndarray) -> bool:
        """
        检查得分是否有效
        
        参数:
            scores: 得分数组
        
        返回:
            bool: 得分是否有效
        """
        if scores is None:
            return False
        if not isinstance(scores, np.ndarray):
            return False
        if len(scores.shape) != 1:
            return False
        # 允许NaN值（某些方法可能产生NaN）
        return not np.all(np.isinf(scores))
    
    @staticmethod
    def measure_execution_time(func, *args, **kwargs) -> Tuple[Any, float]:
        """
        测量函数执行时间
        
        参数:
            func: 要测量的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
        
        返回:
            Tuple: (函数结果, 执行时间)
        """
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            return result, execution_time
        except Exception as e:
            execution_time = time.time() - start_time
            return e, execution_time
    
    @staticmethod
    def capture_method_calls(decision_object, method_names: List[str]) -> Dict[str, bool]:
        """
        检查决策对象的方法是否被成功调用
        
        参数:
            decision_object: 决策对象
            method_names: 要检查的方法名列表
        
        返回:
            Dict: 方法名到调用成功状态的映射
        """
        call_status = {}
        
        for method_name in method_names:
            try:
                if hasattr(decision_object, method_name):
                    method = getattr(decision_object, method_name)
                    if callable(method):
                        # 尝试调用方法（无参数）
                        if method_name in ['get_all_results', 'compare_weights', 'compare_rankings', 'compare_scores']:
                            result = method()
                            call_status[method_name] = result is not None
                        else:
                            call_status[method_name] = True
                    else:
                        call_status[method_name] = False
                else:
                    call_status[method_name] = False
            except Exception:
                call_status[method_name] = False
        
        return call_status
    
    @staticmethod
    def validate_decision_results(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        验证决策结果的有效性
        
        参数:
            results: 决策结果字典
        
        返回:
            Dict: 验证结果
        """
        validation_report = {}
        
        for method_name, result in results.items():
            method_validation = {
                'has_result': result is not None,
                'result_type': type(result).__name__,
                'has_weights': False,
                'has_ranking': False,
                'has_scores': False,
                'weights_valid': False,
                'ranking_valid': False,
                'scores_valid': False
            }
            
            if isinstance(result, dict):
                # 检查权重
                if 'weights' in result:
                    method_validation['has_weights'] = True
                    if isinstance(result['weights'], np.ndarray):
                        method_validation['weights_valid'] = TestHelper.is_valid_weights(result['weights'])
                
                # 检查排名
                if 'ranking' in result:
                    method_validation['has_ranking'] = True
                    if isinstance(result['ranking'], np.ndarray):
                        n_alternatives = len(result['ranking'])
                        method_validation['ranking_valid'] = TestHelper.is_valid_ranking(result['ranking'], n_alternatives)
                
                # 检查得分
                if 'scores' in result:
                    method_validation['has_scores'] = True
                    if isinstance(result['scores'], np.ndarray):
                        method_validation['scores_valid'] = TestHelper.is_valid_scores(result['scores'])
            
            validation_report[method_name] = method_validation
        
        return validation_report
    
    @staticmethod
    def generate_error_report(test_results: Dict[str, Dict]) -> str:
        """
        生成错误报告
        
        参数:
            test_results: 测试结果字典
        
        返回:
            str: 错误报告
        """
        report_lines = ["## 测试错误报告\n"]
        
        has_errors = False
        for test_category, category_results in test_results.items():
            category_errors = []
            
            for test_name, test_result in category_results.items():
                if isinstance(test_result, dict) and test_result.get('status') == 'failed':
                    has_errors = True
                    error_info = test_result.get('error', '未知错误')
                    category_errors.append(f"- **{test_name}**: {error_info}")
            
            if category_errors:
                report_lines.append(f"### {test_category}\n")
                report_lines.extend(category_errors)
                report_lines.append("")
        
        if not has_errors:
            report_lines.append("✅ 所有测试都通过了！没有发现错误。")
        
        return "\n".join(report_lines)
    
    @staticmethod
    def compare_execution_times(times_dict: Dict[str, float], threshold: float = 1.0) -> Dict[str, str]:
        """
        比较执行时间并给出评估
        
        参数:
            times_dict: 方法名到执行时间的映射
            threshold: 时间阈值（秒）
        
        返回:
            Dict: 方法名到性能评估的映射
        """
        performance_report = {}
        
        for method_name, execution_time in times_dict.items():
            if execution_time < 0.01:
                performance_report[method_name] = "极快"
            elif execution_time < 0.1:
                performance_report[method_name] = "快"
            elif execution_time < threshold:
                performance_report[method_name] = "正常"
            elif execution_time < threshold * 3:
                performance_report[method_name] = "较慢"
            else:
                performance_report[method_name] = "很慢"
        
        return performance_report
    
    @staticmethod
    def matrix_properties(matrix: np.ndarray) -> Dict[str, Any]:
        """
        分析矩阵属性
        
        参数:
            matrix: 输入矩阵
        
        返回:
            Dict: 矩阵属性字典
        """
        if matrix is None or matrix.size == 0:
            return {'valid': False, 'reason': '矩阵为空'}
        
        properties = {
            'valid': True,
            'shape': matrix.shape,
            'n_alternatives': matrix.shape[0],
            'n_criteria': matrix.shape[1] if len(matrix.shape) > 1 else 1,
            'has_nan': np.any(np.isnan(matrix)),
            'has_inf': np.any(np.isinf(matrix)),
            'has_negative': np.any(matrix < 0),
            'min_value': np.nanmin(matrix) if not np.all(np.isnan(matrix)) else np.nan,
            'max_value': np.nanmax(matrix) if not np.all(np.isnan(matrix)) else np.nan,
            'mean_value': np.nanmean(matrix) if not np.all(np.isnan(matrix)) else np.nan
        }
        
        return properties


class MethodTester:
    """方法测试器"""
    
    def __init__(self, decision_class):
        """
        初始化方法测试器
        
        参数:
            decision_class: 决策类
        """
        self.decision_class = decision_class
        self.test_results = {}
    
    def test_basic_functionality(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        测试基础功能
        
        参数:
            test_data: 测试数据
        
        返回:
            Dict: 测试结果
        """
        result = {
            'status': 'unknown',
            'error': None,
            'execution_time': 0,
            'methods_executed': 0,
            'results_count': 0
        }
        
        try:
            # 创建决策对象并执行
            decision_obj, execution_time = TestHelper.measure_execution_time(
                self.decision_class
            )
            
            if isinstance(decision_obj, Exception):
                result['status'] = 'failed'
                result['error'] = str(decision_obj)
                return result
            
            # 调用decide方法
            decide_result, decide_time = TestHelper.measure_execution_time(
                decision_obj.decide, **test_data
            )
            
            if isinstance(decide_result, Exception):
                result['status'] = 'failed'
                result['error'] = str(decide_result)
                return result
            
            # 获取结果
            all_results = decision_obj.get_all_results()
            
            result['status'] = 'passed'
            result['execution_time'] = execution_time + decide_time
            result['methods_executed'] = getattr(decision_obj, 'executed_methods', 0)
            result['results_count'] = len(all_results) if all_results else 0
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
        
        return result
    
    def test_boundary_conditions(self, boundary_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        测试边界条件
        
        参数:
            boundary_data: 边界测试数据
        
        返回:
            Dict: 测试结果
        """
        boundary_results = {}
        
        for condition_name, data in boundary_data.items():
            result = {
                'status': 'unknown',
                'error': None,
                'handles_condition': False
            }
            
            try:
                decision_obj = self.decision_class()
                decision_obj.decide(**data)
                
                # 如果没有抛出异常，说明处理了边界条件
                result['status'] = 'passed'
                result['handles_condition'] = True
                
            except Exception as e:
                # 某些边界条件应该抛出异常，这是正常的
                result['status'] = 'handled_with_exception'
                result['error'] = str(e)
                result['handles_condition'] = True
            
            boundary_results[condition_name] = result
        
        return boundary_results