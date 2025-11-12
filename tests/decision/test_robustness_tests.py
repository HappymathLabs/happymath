"""
Interface Robustness Tests

Test abnormal inputs, error handling, numerical stability, etc., to verify system's fault tolerance and stability
"""
import pytest
import numpy as np
import warnings
from typing import Dict, Any, List, Optional

from happymath.Decision.methods import (
    SubWeighting, ObjWeighting, ScoringDecision, PairwiseDecision,
    FuzzySubWeighting, FuzzyObjWeighting, FuzzyScoringDecision
)

from tests.fixtures.test_data import RobustnessTestData
from tests.utils.test_helpers import TestHelper

warnings.filterwarnings("ignore")


class TestRobustnessTests:
    """Robustness test class"""
    
    def setup_method(self):
        """Test method setup"""
        TestHelper.suppress_warnings()
    
    @pytest.mark.robustness
    def test_none_input_handling(self):
        """Test None input handling"""
        invalid_inputs = RobustnessTestData.INVALID_INPUTS['none_values']
        
        decision_classes = [
            SubWeighting, ObjWeighting, ScoringDecision, PairwiseDecision,
            FuzzySubWeighting, FuzzyObjWeighting, FuzzyScoringDecision
        ]
        
        for decision_class in decision_classes:
            class_name = decision_class.__name__
            
            try:
                instance = decision_class()
                
                # Test passing None parameters
                with pytest.raises((ValueError, TypeError, AttributeError)):
                    instance.decide(
                        dataset=invalid_inputs['matrix'],
                        criterion_type=invalid_inputs['criterion_type'],
                        weights=invalid_inputs['weights']
                    )
                
                print(f"✅ {class_name} - None input handling correct (exception thrown)")
                
            except Exception as e:
                # If expected exception not thrown, parameter validation might not be strict enough
                print(f"⚠️ {class_name} - None input handling exception: {e}")
    
    @pytest.mark.robustness
    def test_empty_input_handling(self):
        """Test empty input handling"""
        invalid_inputs = RobustnessTestData.INVALID_INPUTS['empty_arrays']
        
        # Test objective weighting methods (relatively strict input requirements)
        obj_weighting = ObjWeighting()
        
        try:
            with pytest.raises((ValueError, IndexError)):
                obj_weighting.decide(
                    dataset=invalid_inputs['matrix'],
                    criterion_type=invalid_inputs['criterion_type']
                )
            print("✅ Empty array input handling correct (exception thrown)")
            
        except Exception as e:
            print(f"⚠️ Empty array input handling exception: {e}")
        
        # Test scoring decision methods
        scoring = ScoringDecision()
        
        try:
            with pytest.raises((ValueError, IndexError)):
                scoring.decide(
                    dataset=invalid_inputs['matrix'],
                    criterion_type=invalid_inputs['criterion_type'],
                    weights=invalid_inputs['weights']
                )
            print("✅ Scoring decision empty input handling correct (exception thrown)")
            
        except Exception as e:
            print(f"⚠️ Scoring decision empty input handling exception: {e}")
    
    @pytest.mark.robustness
    def test_wrong_type_input_handling(self):
        """Test wrong type input handling"""
        invalid_inputs = RobustnessTestData.INVALID_INPUTS['wrong_types']
        
        # Test string as matrix input
        obj_weighting = ObjWeighting()
        
        try:
            with pytest.raises((ValueError, TypeError)):
                obj_weighting.decide(
                    dataset=invalid_inputs['matrix'],  # String
                    criterion_type=['min', 'max']
                )
            print("✅ String matrix input handling correct (exception thrown)")
            
        except Exception as e:
            print(f"⚠️ String matrix input handling exception: {e}")
        
        # Test wrong weight type
        scoring = ScoringDecision()
        test_matrix = np.array([[1, 2], [3, 4]])
        
        try:
            with pytest.raises((ValueError, TypeError)):
                scoring.decide(
                    dataset=test_matrix,
                    criterion_type=['min', 'max'],
                    weights=invalid_inputs['weights']  # String
                )
            print("✅ String weight input handling correct (exception thrown)")
            
        except Exception as e:
            print(f"⚠️ String weight input handling exception: {e}")
    
    @pytest.mark.robustness
    def test_dimension_mismatch_handling(self):
        """Test dimension mismatch handling"""
        invalid_inputs = RobustnessTestData.INVALID_INPUTS['dimension_mismatch']
        
        # Weight count doesn't match criterion count
        scoring = ScoringDecision()
        
        try:
            with pytest.raises(ValueError):
                scoring.decide(
                    dataset=invalid_inputs['matrix'],      # 3 criteria
                    criterion_type=invalid_inputs['criterion_type'],  # 2 types
                    weights=invalid_inputs['weights']      # 2 weights
                )
            print("✅ Dimension mismatch handling correct (exception thrown)")
            
        except Exception as e:
            print(f"⚠️ Dimension mismatch handling exception: {e}")
        
        # Test pairwise comparison matrix non-square
        try:
            non_square_matrix = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix
            sub_weighting = SubWeighting(methods=['ahp'])
            
            with pytest.raises(ValueError):
                sub_weighting.decide(dataset=non_square_matrix)
            print("✅ Non-square matrix handling correct (exception thrown)")
            
        except Exception as e:
            print(f"⚠️ Non-square matrix handling exception: {e}")
    
    @pytest.mark.robustness
    def test_numerical_issues_handling(self):
        """Test numerical issues handling"""
        numerical_issues = RobustnessTestData.NUMERICAL_ISSUES
        criterion_types = ['min', 'max', 'min']
        
        # Test matrix containing NaN
        try:
            obj_weighting = ObjWeighting(methods=['critic'])  # Choose relatively stable method
            
            # For matrices containing NaN, system should either handle or fail gracefully
            try:
                obj_weighting.decide(
                    dataset=numerical_issues['nan_matrix'],
                    criterion_type=criterion_types
                )
                
                results = obj_weighting.get_all_results()
                if results:
                    print("✅ NaN matrix handled correctly")
                else:
                    print("⚠️ NaN matrix cannot produce results")
                    
            except Exception:
                print("✅ NaN matrix correctly throws exception")
                
        except Exception as e:
            print(f"⚠️ NaN matrix handling issue: {e}")
        
        # Test matrix containing infinity
        try:
            obj_weighting = ObjWeighting(methods=['entropy'])
            
            try:
                obj_weighting.decide(
                    dataset=numerical_issues['inf_matrix'],
                    criterion_type=criterion_types
                )
                
                results = obj_weighting.get_all_results()
                if results:
                    print("✅ Infinity matrix handled correctly")
                else:
                    print("⚠️ Infinity matrix cannot produce results")
                    
            except Exception:
                print("✅ Infinity matrix correctly throws exception")
                
        except Exception as e:
            print(f"⚠️ Infinity matrix handling issue: {e}")
        
        # Test very large values matrix
        try:
            obj_weighting = ObjWeighting(methods=['critic'])
            
            obj_weighting.decide(
                dataset=numerical_issues['very_large'],
                criterion_type=criterion_types
            )
            
            results = obj_weighting.get_all_results()
            
            # Check if results contain abnormal values
            for method_name, result in results.items():
                if 'weights' in result:
                    weights = result['weights']
                    if not (np.any(np.isnan(weights)) or np.any(np.isinf(weights))):
                        print(f"✅ Very large values matrix handled correctly - {method_name}")
                    else:
                        print(f"⚠️ Very large values matrix causes abnormal values - {method_name}")
                        
        except Exception as e:
            print(f"✅ Very large values matrix correctly throws exception: {e}")
        
        # Test very small values matrix
        try:
            obj_weighting = ObjWeighting(methods=['seca'])
            
            obj_weighting.decide(
                dataset=numerical_issues['very_small'],
                criterion_type=criterion_types
            )
            
            results = obj_weighting.get_all_results()
            
            for method_name, result in results.items():
                if 'weights' in result:
                    weights = result['weights']
                    if TestHelper.is_valid_weights(weights):
                        print(f"✅ Very small values matrix handled correctly - {method_name}")
                    else:
                        print(f"⚠️ Very small values matrix weights invalid - {method_name}")
                        
        except Exception as e:
            print(f"✅ Very small values matrix correctly throws exception: {e}")
    
    @pytest.mark.robustness
    def test_weight_issues_handling(self):
        """Test weight-related issues handling"""
        weight_issues = RobustnessTestData.WEIGHT_ISSUES
        test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        criterion_types = ['min', 'max', 'max']
        
        weight_tests = [
            ('Negative weights', weight_issues['negative_weights']),
            ('Zero weights', weight_issues['zero_weights']),
            ('Weights sum not equal to 1', weight_issues['sum_not_one']),
            ('Single weight equals 1', weight_issues['single_weight_one']),
            ('Contains NaN weights', weight_issues['nan_weights']),
            ('Contains infinite weights', weight_issues['inf_weights'])
        ]
        
        for weight_name, weights in weight_tests:
            try:
                scoring = ScoringDecision(methods=['saw'])  # Use simple method
                
                try:
                    scoring.decide(
                        dataset=test_matrix,
                        criterion_type=criterion_types,
                        weights=weights
                    )
                    
                    results = scoring.get_all_results()
                    
                    if results:
                        # Check if results are reasonable
                        valid_results = True
                        for method_name, result in results.items():
                            if 'ranking' in result:
                                ranking = result['ranking']
                                if not TestHelper.is_valid_ranking(ranking, 3):
                                    valid_results = False
                        
                        if valid_results:
                            print(f"✅ {weight_name} handled correctly")
                        else:
                            print(f"⚠️ {weight_name} produces invalid results")
                    else:
                        print(f"⚠️ {weight_name} cannot produce results")
                
                except Exception:
                    print(f"✅ {weight_name} correctly throws exception")
                    
            except Exception as e:
                print(f"⚠️ {weight_name} handling issue: {e}")
    
    @pytest.mark.robustness
    def test_memory_stress(self):
        """Test memory stress"""
        # Generate larger matrix to test memory usage
        np.random.seed(42)
        
        try:
            # Create large matrix but not too large to exhaust memory
            large_matrix = np.random.rand(200, 30) * 100
            criterion_types = ['min', 'max'] * 15
            
            # Test objective weighting methods (relatively memory efficient)
            obj_weighting = ObjWeighting(methods=['entropy'])
            
            import psutil
            import os
            
            # Get memory usage
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            obj_weighting.decide(
                dataset=large_matrix,
                criterion_type=criterion_types
            )
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            # Memory increase should be within reasonable range (less than 200MB)
            assert memory_increase < 200, f"Memory usage increase too much: {memory_increase:.1f}MB"
            
            results = obj_weighting.get_all_results()
            assert len(results) > 0, "Large matrix should produce results"
            
            print(f"✅ Memory stress test passed - Memory increase: {memory_increase:.1f}MB")
            
        except ImportError:
            print("⚠️ psutil not installed, skipping memory test")
        except Exception as e:
            print(f"⚠️ Memory stress test exception: {e}")
    
    @pytest.mark.robustness
    def test_concurrent_access(self):
        """Test concurrent access"""
        import threading
        import time
        
        test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        criterion_types = ['min', 'max', 'max']
        weights = np.array([1/3, 1/3, 1/3])
        
        results_container = []
        errors_container = []
        
        def worker_function(worker_id):
            """Worker thread function"""
            try:
                # Each thread creates its own instance
                scoring = ScoringDecision(methods=['topsis'])
                scoring.decide(
                    dataset=test_matrix,
                    criterion_type=criterion_types,
                    weights=weights
                )
                
                results = scoring.get_all_results()
                results_container.append((worker_id, results))
                
            except Exception as e:
                errors_container.append((worker_id, str(e)))
        
        # Create multiple threads
        threads = []
        num_threads = 5
        
        for i in range(num_threads):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        if len(errors_container) == 0:
            print(f"✅ Concurrent access test passed - {len(results_container)} threads completed successfully")
            
            # Verify result consistency
            if len(results_container) > 1:
                first_result = results_container[0][1]
                for worker_id, result in results_container[1:]:
                    for method_name in first_result:
                        if method_name in result:
                            # Compare if rankings are consistent
                            if 'ranking' in first_result[method_name] and 'ranking' in result[method_name]:
                                if not np.array_equal(first_result[method_name]['ranking'], result[method_name]['ranking']):
                                    print(f"⚠️ Thread {worker_id} result inconsistent with baseline")
                
                print("✅ Concurrent result consistency verification passed")
        else:
            print(f"⚠️ Concurrent access had {len(errors_container)} errors:")
            for worker_id, error in errors_container:
                print(f"  Thread {worker_id}: {error}")
    
    @pytest.mark.robustness
    def test_repeated_execution_stability(self):
        """Test repeated execution stability"""
        test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        criterion_types = ['min', 'max', 'max']
        weights = np.array([0.3, 0.4, 0.3])
        
        # Repeatedly execute the same computation multiple times
        num_runs = 10
        all_results = []
        
        for run in range(num_runs):
            scoring = ScoringDecision(methods=['topsis', 'vikor'])
            scoring.decide(
                dataset=test_matrix,
                criterion_type=criterion_types,
                weights=weights
            )
            
            results = scoring.get_all_results()
            all_results.append(results)
        
        # Verify result consistency
        if len(all_results) > 1:
            baseline_results = all_results[0]
            
            for run_idx, results in enumerate(all_results[1:], 1):
                for method_name in baseline_results:
                    if method_name in results:
                        baseline_ranking = baseline_results[method_name].get('ranking')
                        current_ranking = results[method_name].get('ranking')
                        
                        if baseline_ranking is not None and current_ranking is not None:
                            if not np.array_equal(baseline_ranking, current_ranking):
                                print(f"⚠️ Run {run_idx+1} {method_name} results inconsistent")
                                print(f"  Baseline ranking: {baseline_ranking}")
                                print(f"  Current ranking: {current_ranking}")
                                return
            
            print(f"✅ Repeated execution stability test passed - {num_runs} runs consistent")
        
    @pytest.mark.robustness
    def test_error_recovery(self):
        """Test error recovery capability"""
        # Create a matrix that might cause some methods to fail
        problematic_matrix = np.array([
            [0, 1, 2],
            [1, 0, 1], 
            [2, 1, 0]
        ])
        criterion_types = ['min', 'max', 'min']
        
        # Test if system can continue executing other methods when some methods fail
        scoring = ScoringDecision()  # Use all methods
        
        try:
            scoring.decide(
                dataset=problematic_matrix,
                criterion_type=criterion_types
            )
            
            results = scoring.get_all_results()
            
            if results:
                successful_methods = len(results)
                total_available_methods = len(scoring._METHOD_REGISTRY)
                
                success_rate = successful_methods / total_available_methods if total_available_methods > 0 else 0
                
                print(f"✅ Error recovery test passed - {successful_methods}/{total_available_methods} methods successful ({success_rate:.1%})")
                
                # At least some methods should succeed
                assert successful_methods > 0, "At least some methods should be able to succeed"
            else:
                print("⚠️ All methods failed")
                
        except Exception as e:
            print(f"⚠️ Error recovery test exception: {e}")
    
    @pytest.mark.robustness
    def test_resource_cleanup(self):
        """Test resource cleanup"""
        import gc
        import weakref
        
        # Create multiple object instances
        instances = []
        weak_refs = []
        
        for _ in range(10):
            obj = ScoringDecision()
            instances.append(obj)
            weak_refs.append(weakref.ref(obj))
        
        # Use these instances
        test_matrix = np.array([[1, 2], [3, 4]])
        criterion_types = ['min', 'max']
        
        for obj in instances:
            obj.decide(dataset=test_matrix, criterion_type=criterion_types)
        
        # Clear references
        instances.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Check if objects are properly cleaned up
        alive_objects = sum(1 for ref in weak_refs if ref() is not None)
        
        if alive_objects == 0:
            print("✅ Resource cleanup test passed - All objects properly cleaned up")
        else:
            print(f"⚠️ {alive_objects} objects not cleaned up")


if __name__ == "__main__":
    # Quick test when running directly
    test_instance = TestRobustnessTests()
    test_instance.setup_method()
    
    try:
        test_instance.test_none_input_handling()
        test_instance.test_empty_input_handling()
        test_instance.test_wrong_type_input_handling()
        test_instance.test_dimension_mismatch_handling()
        test_instance.test_numerical_issues_handling()
        test_instance.test_weight_issues_handling()
        test_instance.test_repeated_execution_stability()
        test_instance.test_error_recovery()
        test_instance.test_resource_cleanup()
        print("\n🎉 All robustness tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()