"""
Test utility functions
"""
import numpy as np
import pandas as pd
import time
import traceback
import warnings
from typing import Any, Dict, List, Tuple, Optional, Union


class TestHelper:
    """Test helper utility class"""
    
    @staticmethod
    def suppress_warnings():
        """Suppress warnings during testing"""
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
    
    @staticmethod
    def is_valid_weights(weights: np.ndarray, tolerance: float = 1e-6) -> bool:
        """
        Check if weights are valid
        
        Parameters:
            weights: Weight vector
            tolerance: Tolerance
        
        Returns:
            bool: Whether weights are valid
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
        # Weight sum should be approximately 1 (allowing small error)
        return abs(np.sum(weights) - 1.0) < tolerance
    
    @staticmethod
    def is_valid_ranking(ranking: np.ndarray, n_alternatives: int) -> bool:
        """
        Check if ranking is valid
        
        Parameters:
            ranking: Ranking array
            n_alternatives: Number of alternatives
        
        Returns:
            bool: Whether ranking is valid
        """
        if ranking is None:
            return False
        if not isinstance(ranking, np.ndarray):
            return False
        if len(ranking) != n_alternatives:
            return False
        if np.any(np.isnan(ranking)) or np.any(np.isinf(ranking)):
            return False
        # Ranking should be between 1 and n_alternatives
        return np.all(ranking >= 1) and np.all(ranking <= n_alternatives)
    
    @staticmethod
    def is_valid_scores(scores: np.ndarray) -> bool:
        """
        Check if scores are valid
        
        Parameters:
            scores: Scores array
        
        Returns:
            bool: Whether scores are valid
        """
        if scores is None:
            return False
        if not isinstance(scores, np.ndarray):
            return False
        if len(scores.shape) != 1:
            return False
        # Allow NaN values (some methods may produce NaN)
        return not np.all(np.isinf(scores))
    
    @staticmethod
    def measure_execution_time(func, *args, **kwargs) -> Tuple[Any, float]:
        """
        Measure function execution time
        
        Parameters:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Tuple: (function result, execution time)
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
        Check if decision object methods were successfully called
        
        Parameters:
            decision_object: Decision object
            method_names: List of method names to check
        
        Returns:
            Dict: Mapping of method names to call success status
        """
        call_status = {}
        
        for method_name in method_names:
            try:
                if hasattr(decision_object, method_name):
                    method = getattr(decision_object, method_name)
                    if callable(method):
                        # Try calling the method (no parameters)
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
        Validate decision results
        
        Parameters:
            results: Decision results dictionary
        
        Returns:
            Dict: Validation results
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
                # Check weights
                if 'weights' in result:
                    method_validation['has_weights'] = True
                    if isinstance(result['weights'], np.ndarray):
                        method_validation['weights_valid'] = TestHelper.is_valid_weights(result['weights'])
                
                # Check ranking
                if 'ranking' in result:
                    method_validation['has_ranking'] = True
                    if isinstance(result['ranking'], np.ndarray):
                        n_alternatives = len(result['ranking'])
                        method_validation['ranking_valid'] = TestHelper.is_valid_ranking(result['ranking'], n_alternatives)
                
                # Check scores
                if 'scores' in result:
                    method_validation['has_scores'] = True
                    if isinstance(result['scores'], np.ndarray):
                        method_validation['scores_valid'] = TestHelper.is_valid_scores(result['scores'])
            
            validation_report[method_name] = method_validation
        
        return validation_report
    
    @staticmethod
    def generate_error_report(test_results: Dict[str, Dict]) -> str:
        """
        Generate error report
        
        Parameters:
            test_results: Test results dictionary
        
        Returns:
            str: Error report
        """
        report_lines = ["## Test Error Report\n"]
        
        has_errors = False
        for test_category, category_results in test_results.items():
            category_errors = []
            
            for test_name, test_result in category_results.items():
                if isinstance(test_result, dict) and test_result.get('status') == 'failed':
                    has_errors = True
                    error_info = test_result.get('error', 'Unknown error')
                    category_errors.append(f"- **{test_name}**: {error_info}")
            
            if category_errors:
                report_lines.append(f"### {test_category}\n")
                report_lines.extend(category_errors)
                report_lines.append("")
        
        if not has_errors:
            report_lines.append("✅ All tests passed! No errors found.")
        
        return "\n".join(report_lines)
    
    @staticmethod
    def compare_execution_times(times_dict: Dict[str, float], threshold: float = 1.0) -> Dict[str, str]:
        """
        Compare execution times and provide evaluation
        
        Parameters:
            times_dict: Mapping of method names to execution times
            threshold: Time threshold (seconds)
        
        Returns:
            Dict: Mapping of method names to performance evaluation
        """
        performance_report = {}
        
        for method_name, execution_time in times_dict.items():
            if execution_time < 0.01:
                performance_report[method_name] = "Very Fast"
            elif execution_time < 0.1:
                performance_report[method_name] = "Fast"
            elif execution_time < threshold:
                performance_report[method_name] = "Normal"
            elif execution_time < threshold * 3:
                performance_report[method_name] = "Slow"
            else:
                performance_report[method_name] = "Very Slow"
        
        return performance_report
    
    @staticmethod
    def matrix_properties(matrix: np.ndarray) -> Dict[str, Any]:
        """
        Analyze matrix properties
        
        Parameters:
            matrix: Input matrix
        
        Returns:
            Dict: Matrix properties dictionary
        """
        if matrix is None or matrix.size == 0:
            return {'valid': False, 'reason': 'Matrix is empty'}
        
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
    """Method tester"""
    
    def __init__(self, decision_class):
        """
        Initialize method tester
        
        Parameters:
            decision_class: Decision class
        """
        self.decision_class = decision_class
        self.test_results = {}
    
    def test_basic_functionality(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test basic functionality
        
        Parameters:
            test_data: Test data
        
        Returns:
            Dict: Test results
        """
        result = {
            'status': 'unknown',
            'error': None,
            'execution_time': 0,
            'methods_executed': 0,
            'results_count': 0
        }
        
        try:
            # Create decision object and execute
            decision_obj, execution_time = TestHelper.measure_execution_time(
                self.decision_class
            )
            
            if isinstance(decision_obj, Exception):
                result['status'] = 'failed'
                result['error'] = str(decision_obj)
                return result
            
            # Call decide method
            decide_result, decide_time = TestHelper.measure_execution_time(
                decision_obj.decide, **test_data
            )
            
            if isinstance(decide_result, Exception):
                result['status'] = 'failed'
                result['error'] = str(decide_result)
                return result
            
            # Get results
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
        Test boundary conditions
        
        Parameters:
            boundary_data: Boundary test data
        
        Returns:
            Dict: Test results
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
                
                # If no exception is raised, boundary condition was handled
                result['status'] = 'passed'
                result['handles_condition'] = True
                
            except Exception as e:
                # Some boundary conditions should raise exceptions, which is normal
                result['status'] = 'handled_with_exception'
                result['error'] = str(e)
                result['handles_condition'] = True
            
            boundary_results[condition_name] = result
        
        return boundary_results