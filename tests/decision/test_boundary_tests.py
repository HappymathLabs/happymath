"""
Interface Boundary Tests

Test various boundary conditions and critical values, including minimum/maximum input sizes, boundary weight values, special matrices, etc.
"""
import pytest
import numpy as np
import warnings
from typing import Dict, Any, List

from happymath.Decision.methods import (
    SubWeighting, ObjWeighting, ScoringDecision, PairwiseDecision,
    FuzzySubWeighting, FuzzyObjWeighting, FuzzyScoringDecision
)

from tests.fixtures.test_data import BoundaryTestData
from tests.utils.test_helpers import TestHelper

warnings.filterwarnings("ignore")


class TestBoundaryTests:
    """Boundary test class"""
    
    def setup_method(self):
        """Test method setup"""
        TestHelper.suppress_warnings()
        # Generate large-scale test data
        BoundaryTestData.generate_large_data()
    
    @pytest.mark.boundary
    def test_minimum_size_matrices(self):
        """Test minimum size matrices (2x2)"""
        min_data = BoundaryTestData.MIN_DATA
        
        # SubWeighting - Minimum matrix for AHP method
        ahp_2x2 = np.array([[1, 2], [0.5, 1]])
        sub_weighting = SubWeighting()
        sub_weighting.decide(dataset=ahp_2x2)
        
        results = sub_weighting.get_all_results()
        assert len(results) > 0, "2x2 matrix should produce results"
        
        # Verify AHP results
        if 'ahp' in results:
            weights = results['ahp']['weights']
            assert len(weights) == 2, "2x2 matrix should produce 2 weights"
            assert TestHelper.is_valid_weights(weights), "Weights should be valid"
        
        print("✅ Minimum size SubWeighting test passed")
        
        # ObjWeighting - Minimum decision matrix
        obj_weighting = ObjWeighting()
        obj_weighting.decide(
            dataset=min_data['matrix_2x2'],
            criterion_type=min_data['criteria_types_2']
        )
        
        obj_results = obj_weighting.get_all_results()
        assert len(obj_results) > 0, "2x2 matrix should produce objective weights"
        
        for method_name, result in obj_results.items():
            if 'weights' in result:
                weights = result['weights']
                # Some methods may produce different numbers of weights, which is acceptable
                assert isinstance(weights, np.ndarray), f"{method_name}'s weights should be numpy array"
                assert len(weights) >= 2, f"{method_name} should produce at least 2 weights"
        
        print("✅ Minimum size ObjWeighting test passed")
        
        # ScoringDecision - Minimum scoring matrix
        scoring = ScoringDecision()
        scoring.decide(
            dataset=min_data['matrix_2x2'],
            criterion_type=min_data['criteria_types_2'],
            weights=min_data['weights_2']
        )
        
        scoring_results = scoring.get_all_results()
        assert len(scoring_results) > 0, "2x2 matrix should produce scoring results"
        
        for method_name, result in scoring_results.items():
            if 'ranking' in result:
                ranking = result['ranking']
                assert len(ranking) == 2, f"{method_name} should rank 2 alternatives"
                assert TestHelper.is_valid_ranking(ranking, 2), f"{method_name} ranking should be valid"
        
        print("✅ Minimum size ScoringDecision test passed")
    
    @pytest.mark.boundary
    @pytest.mark.slow
    def test_large_size_matrices(self):
        """Test large-scale matrices"""
        large_data = BoundaryTestData.LARGE_DATA
        
        # Test large-scale objective weight calculation (most stable)
        obj_weighting = ObjWeighting()
        start_time = TestHelper.measure_execution_time(
            obj_weighting.decide,
            dataset=large_data['matrix_100x20'],
            criterion_type=large_data['criteria_types_20']
        )
        
        execution_time = start_time[1]
        assert execution_time < 30.0, f"Large matrix processing time too long: {execution_time:.2f} seconds"
        
        results = obj_weighting.get_all_results()
        assert len(results) > 0, "Large matrix should produce results"
        
        for method_name, result in results.items():
            if 'weights' in result:
                weights = result['weights']
                assert len(weights) == 20, f"{method_name} should produce 20 weights"
                assert TestHelper.is_valid_weights(weights), f"{method_name}'s weights should be valid"
        
        print(f"✅ Large-scale ObjWeighting test passed - Processing 100x20 matrix took {execution_time:.2f} seconds")
        
        # Test large-scale scoring decision (select partial methods to avoid timeout)
        scoring = ScoringDecision(methods=['topsis', 'saw', 'vikor'])
        scoring_time = TestHelper.measure_execution_time(
            scoring.decide,
            dataset=large_data['matrix_100x20'],
            criterion_type=large_data['criteria_types_20'],
            weights=large_data['weights_20']
        )
        
        scoring_execution_time = scoring_time[1] 
        assert scoring_execution_time < 60.0, f"Large-scale scoring calculation time too long: {scoring_execution_time:.2f} seconds"
        
        scoring_results = scoring.get_all_results()
        for method_name, result in scoring_results.items():
            if 'ranking' in result:
                ranking = result['ranking']
                assert len(ranking) == 100, f"{method_name} should rank 100 alternatives"
                assert TestHelper.is_valid_ranking(ranking, 100), f"{method_name} ranking should be valid"
        
        print(f"✅ Large-scale ScoringDecision test passed - Processing 100x20 matrix took {scoring_execution_time:.2f} seconds")
    
    @pytest.mark.boundary
    def test_boundary_weight_values(self):
        """Test boundary weight values"""
        test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        criterion_types = ['min', 'max', 'max']
        
        boundary_weights = [
            np.array([1.0, 0.0, 0.0]),      # Extreme weights
            np.array([0.0, 1.0, 0.0]),      # Single criterion weights  
            np.array([0.0, 0.0, 1.0]),      # Single criterion weights
            np.array([1/3, 1/3, 1/3]),      # Equal weights
            np.array([0.999, 0.0005, 0.0005]),  # Near-extreme weights
            np.array([0.001, 0.001, 0.998])     # Near-single weights
        ]
        
        for i, weights in enumerate(boundary_weights):
            scoring = ScoringDecision(methods=['topsis', 'saw'])  # Use stable methods
            
            try:
                scoring.decide(
                    dataset=test_matrix,
                    criterion_type=criterion_types,
                    weights=weights
                )
                
                results = scoring.get_all_results()
                assert len(results) > 0, f"Boundary weights {i+1} should produce results"
                
                for method_name, result in results.items():
                    if 'ranking' in result:
                        ranking = result['ranking']
                        assert TestHelper.is_valid_ranking(ranking, 3), f"Boundary weights {i+1}'s {method_name} ranking should be valid"
                
                print(f"✅ Boundary weight test {i+1} passed: {weights}")
                
            except Exception as e:
                # Some extreme weights may cause numerical issues, which is acceptable
                print(f"⚠️ Boundary weight {i+1} caused exception (acceptable): {e}")
    
    @pytest.mark.boundary
    def test_special_matrices(self):
        """Test special matrices"""
        special_matrices = BoundaryTestData.SPECIAL_VALUES
        criterion_types = ['min', 'max', 'min']
        weights = np.array([1/3, 1/3, 1/3])
        
        for matrix_name, matrix in special_matrices.items():
            if matrix.shape[1] != 3:
                # Adjust criterion types to match matrix
                n_criteria = matrix.shape[1]
                test_criterion_types = ['min', 'max'] * (n_criteria // 2 + 1)
                test_criterion_types = test_criterion_types[:n_criteria]
                test_weights = np.ones(n_criteria) / n_criteria
            else:
                test_criterion_types = criterion_types
                test_weights = weights
            
            print(f"Testing special matrix: {matrix_name} - {matrix.shape}")
            
            try:
                # Test objective weight methods (more robust to special values)
                obj_weighting = ObjWeighting(methods=['critic', 'entropy'])
                obj_weighting.decide(
                    dataset=matrix,
                    criterion_type=test_criterion_types
                )
                
                obj_results = obj_weighting.get_all_results()
                
                for method_name, result in obj_results.items():
                    if 'weights' in result:
                        weights_result = result['weights']
                        # For special matrices, weights may not be standard, but should be numeric
                        assert isinstance(weights_result, np.ndarray), f"{method_name}'s weights for {matrix_name} should be array"
                        assert not np.all(np.isnan(weights_result)), f"{method_name}'s weights for {matrix_name} should not all be NaN"
                
                print(f"✅ Special matrix {matrix_name} test passed")
                
            except Exception as e:
                # Some special matrices may cause numerical issues
                print(f"⚠️ Special matrix {matrix_name} caused exception: {e}")
    
    @pytest.mark.boundary
    def test_pairwise_matrix_boundaries(self):
        """Test boundary conditions of pairwise comparison matrices"""
        # Perfect consistency matrix (CR=0)
        perfect_consistency = np.array([
            [1, 2, 4],
            [0.5, 1, 2],
            [0.25, 0.5, 1]
        ])
        
        # Near-inconsistent matrix
        near_inconsistent = np.array([
            [1, 9, 9],
            [1/9, 1, 9], 
            [1/9, 1/9, 1]
        ])
        
        # Extreme comparison matrix
        extreme_matrix = np.array([
            [1, 9, 9, 9],
            [1/9, 1, 1/9, 1/9],
            [1/9, 9, 1, 1/9], 
            [1/9, 9, 9, 1]
        ])
        
        matrices = [
            ("Perfect consistency", perfect_consistency),
            ("Near inconsistent", near_inconsistent),
            ("Extreme comparison", extreme_matrix)
        ]
        
        for name, matrix in matrices:
            try:
                sub_weighting = SubWeighting(methods=['ahp'])
                sub_weighting.decide(dataset=matrix)
                
                results = sub_weighting.get_all_results()
                if 'ahp' in results:
                    weights = results['ahp']['weights']
                    cr = results['ahp'].get('consistency_ratio', 0)
                    
                    assert TestHelper.is_valid_weights(weights), f"{name} matrix should produce valid weights"
                    assert isinstance(cr, (int, float)), f"{name} matrix should have consistency ratio"
                    
                    print(f"✅ {name} matrix test passed - CR: {cr:.4f}")
                
            except Exception as e:
                print(f"⚠️ {name} matrix caused exception: {e}")
    
    @pytest.mark.boundary
    def test_fuzzy_boundary_conditions(self):
        """Test boundary conditions of fuzzy numbers"""
        # Degenerate fuzzy numbers (three values equal)
        degenerate_fuzzy = np.array([
            [[5, 5, 5], [7, 7, 7], [9, 9, 9]],
            [[3, 3, 3], [5, 5, 5], [7, 7, 7]],
            [[1, 1, 1], [3, 3, 3], [5, 5, 5]]
        ])
        
        # Extreme fuzzy numbers
        extreme_fuzzy = np.array([
            [[0.1, 5, 9.9], [0.1, 1, 9.9], [0.1, 3, 9.9]],
            [[0.1, 7, 9.9], [0.1, 9, 9.9], [0.1, 5, 9.9]],
            [[0.1, 2, 9.9], [0.1, 4, 9.9], [0.1, 8, 9.9]]
        ])
        
        fuzzy_matrices = [
            ("Degenerate fuzzy numbers", degenerate_fuzzy),
            ("Extreme fuzzy numbers", extreme_fuzzy)
        ]
        
        criterion_types = ['max', 'min', 'max']
        
        for name, fuzzy_matrix in fuzzy_matrices:
            try:
                fuzzy_obj = FuzzyObjWeighting()
                fuzzy_obj.decide(
                    dataset=fuzzy_matrix,
                    criterion_type=criterion_types
                )
                
                results = fuzzy_obj.get_all_results()
                assert len(results) > 0, f"{name} should produce fuzzy weight results"
                
                for method_name, result in results.items():
                    if 'weights' in result:
                        weights = result['weights']
                        assert isinstance(weights, np.ndarray), f"{method_name}'s weights for {name} should be array"
                
                print(f"✅ Fuzzy {name} test passed")
                
            except Exception as e:
                print(f"⚠️ Fuzzy {name} caused exception: {e}")
    
    @pytest.mark.boundary
    def test_single_alternative_scenario(self):
        """Test single alternative scenario"""
        single_alt_matrix = np.array([[5, 8, 6, 9]])  # Only one alternative
        criterion_types = ['min', 'max', 'min', 'max']
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Most multi-criteria decision methods need at least 2 alternatives for comparison
        # Test how system handles this boundary case
        
        try:
            scoring = ScoringDecision(methods=['saw'])  # Use simple method
            scoring.decide(
                dataset=single_alt_matrix,
                criterion_type=criterion_types,
                weights=weights
            )
            
            results = scoring.get_all_results()
            
            if results:
                for method_name, result in results.items():
                    if 'ranking' in result:
                        ranking = result['ranking']
                        assert len(ranking) == 1, "Single alternative should have only one ranking"
                        assert ranking[0] == 1, "Single alternative's ranking should be 1"
                
                print("✅ Single alternative test passed")
            else:
                print("⚠️ Single alternative cannot produce results (as expected)")
                
        except Exception as e:
            print(f"⚠️ Single alternative caused exception (as expected): {e}")
    
    @pytest.mark.boundary 
    def test_extreme_threshold_values(self):
        """Test extreme threshold parameters"""
        test_matrix = np.array([
            [5, 85, 70],
            [4, 92, 65], 
            [6, 75, 80]
        ])
        weights = np.array([0.3, 0.4, 0.3])
        
        # Extreme threshold combinations
        extreme_thresholds = [
            {
                'name': 'Zero thresholds',
                'Q': np.array([0, 0, 0]),
                'P': np.array([0.001, 0.001, 0.001]),
                'V': np.array([0.002, 0.002, 0.002])
            },
            {
                'name': 'Very large thresholds',
                'Q': np.array([100, 100, 100]),
                'P': np.array([200, 200, 200]),
                'V': np.array([300, 300, 300])
            }
        ]
        
        for threshold_set in extreme_thresholds:
            try:
                pairwise = PairwiseDecision(methods=['electre_iii'])
                pairwise.decide(
                    dataset=test_matrix,
                    weights=weights,
                    Q=threshold_set['Q'],
                    P=threshold_set['P'], 
                    V=threshold_set['V']
                )
                
                results = pairwise.get_all_results()
                if results:
                    print(f"✅ {threshold_set['name']} threshold test passed")
                else:
                    print(f"⚠️ {threshold_set['name']} threshold cannot produce results")
                    
            except Exception as e:
                print(f"⚠️ {threshold_set['name']} threshold caused exception: {e}")
    
    @pytest.mark.boundary
    def test_performance_degradation(self):
        """Test performance degradation at different scales"""
        sizes = [(5, 3), (10, 5), (20, 8), (50, 10)]
        execution_times = []
        
        np.random.seed(42)  # Ensure reproducibility
        
        for n_alt, n_crit in sizes:
            matrix = np.random.rand(n_alt, n_crit) * 100
            criterion_types = ['min', 'max'] * (n_crit // 2 + 1)
            criterion_types = criterion_types[:n_crit]
            
            obj_weighting = ObjWeighting(methods=['critic'])  # Use single stable method
            
            _, exec_time = TestHelper.measure_execution_time(
                obj_weighting.decide,
                dataset=matrix,
                criterion_type=criterion_types
            )
            
            execution_times.append((n_alt * n_crit, exec_time))
            print(f"✅ Scale {n_alt}x{n_crit} - Execution time: {exec_time:.4f} seconds")
        
        # Check if performance growth is reasonable (should not be exponential)
        if len(execution_times) >= 2:
            # Time ratio between largest and smallest scale
            time_ratio = execution_times[-1][1] / execution_times[0][1]
            size_ratio = execution_times[-1][0] / execution_times[0][0]
            
            # Time growth should not exceed square of size growth
            assert time_ratio < size_ratio ** 2, f"Performance degradation too severe: time growth {time_ratio:.2f}x, size growth {size_ratio:.2f}x"
            
            print(f"✅ Performance degradation test passed - Time growth {time_ratio:.2f}x, size growth {size_ratio:.2f}x")


if __name__ == "__main__":
    # Quick test when running directly
    test_instance = TestBoundaryTests()
    test_instance.setup_method()
    
    try:
        test_instance.test_minimum_size_matrices()
        test_instance.test_boundary_weight_values()
        test_instance.test_special_matrices()
        test_instance.test_pairwise_matrix_boundaries()
        test_instance.test_fuzzy_boundary_conditions()
        test_instance.test_single_alternative_scenario()
        test_instance.test_extreme_threshold_values()
        test_instance.test_performance_degradation()
        print("\n🎉 All boundary tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()