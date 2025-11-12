"""
Functional Smoke Tests

Reference test_all_decision.ipynb sample, test the basic functionality of 7 core decision classes
Ensure all methods can execute normally and all public interfaces are called
"""
import pytest
import numpy as np
import warnings
from typing import Dict, Any

# Import all decision classes
from happymath.Decision.methods import (
    SubWeighting, ObjWeighting, ScoringDecision, PairwiseDecision,
    FuzzySubWeighting, FuzzyObjWeighting, FuzzyScoringDecision
)

from tests.fixtures.test_data import TestData
from tests.utils.test_helpers import TestHelper, MethodTester

# Suppress warnings
warnings.filterwarnings("ignore")


class TestSmokeTests:
    """Functional smoke test class"""
    
    def setup_method(self):
        """Test method setup"""
        TestHelper.suppress_warnings()
    
    @pytest.mark.smoke
    def test_sub_weighting_smoke(self):
        """SubWeighting functional smoke test"""
        # Use test data
        data = TestData.SUB_WEIGHTING_DATA
        
        # Create SubWeighting instance
        sub_weighting = SubWeighting()
        
        # Call decide method with all parameters
        sub_weighting.decide(
            dataset=data['ahp_matrix'],
            mic=data['bwm_data']['mic'],
            lic=data['bwm_data']['lic'],
            criteria_rank=data['fucom_data']['criteria_rank'],
            criteria_priority=data['fucom_data']['criteria_priority']
        )
        
        # Verify basic functionality
        executed_methods = sub_weighting.get_executed_methods()
        assert len(executed_methods) > 0, "Should execute at least one method"
        
        # Verify result retrieval
        all_results = sub_weighting.get_all_results()
        assert isinstance(all_results, dict), "get_all_results should return dict"
        assert len(all_results) > 0, "Should have results"
        
        # Verify weight retrieval
        weights_comparison = sub_weighting.compare_weights()
        assert weights_comparison is not None, "compare_weights should have return value"
        
        # Verify weight result validity
        for method_name, result in all_results.items():
            if isinstance(result, dict) and 'weights' in result:
                weights = result['weights']
                assert isinstance(weights, np.ndarray), f"{method_name} weights should be numpy array"
                assert TestHelper.is_valid_weights(weights), f"{method_name} weights should be valid"
        
        print(f"✅ SubWeighting test passed - executed {len(executed_methods)} methods")
    
    @pytest.mark.smoke  
    def test_obj_weighting_smoke(self):
        """ObjWeighting functional smoke test"""
        # Use test data
        data = TestData.OBJ_WEIGHTING_DATA
        
        # Create ObjWeighting instance
        obj_weighting = ObjWeighting()
        
        # Call decide method
        obj_weighting.decide(
            dataset=data['decision_matrix'],
            criterion_type=data['criterion_types']
        )
        
        # Verify basic functionality
        executed_methods = obj_weighting.get_executed_methods()
        assert len(executed_methods) > 0, "Should execute at least one method"
        
        # Verify result retrieval
        all_results = obj_weighting.get_all_results()
        assert isinstance(all_results, dict), "get_all_results should return dict"
        assert len(all_results) > 0, "Should have results"
        
        # Verify weight comparison
        weights_comparison = obj_weighting.compare_weights()
        assert weights_comparison is not None, "compare_weights should have return value"
        
        # Verify weight validity
        for method_name, result in all_results.items():
            if isinstance(result, dict) and 'weights' in result:
                weights = result['weights']
                assert isinstance(weights, np.ndarray), f"{method_name} weights should be numpy array"
                assert TestHelper.is_valid_weights(weights), f"{method_name} weights should be valid"
        
        print(f"✅ ObjWeighting test passed - executed {len(executed_methods)} methods")
    
    @pytest.mark.smoke
    def test_scoring_decision_smoke(self):
        """ScoringDecision functional smoke test"""
        # Use test data
        data = TestData.SCORING_DATA
        
        # Create ScoringDecision instance
        scoring = ScoringDecision()
        
        # Call decide method with complete parameters
        scoring.decide(
            dataset=data['decision_matrix'],
            criterion_type=data['criterion_types'],
            weights=data['weights'],
            lambda_value=data['waspas_params']['lambda_value'],
            s_min=data['spotis_params']['s_min'],
            s_max=data['spotis_params']['s_max'],
            grades=data['smart_params']['grades'],
            lower=data['smart_params']['lower'],
            upper=data['smart_params']['upper'],
            utility_functions=data['smart_params']['utility_functions']
        )
        
        # Verify basic functionality
        executed_methods = scoring.get_executed_methods()
        assert len(executed_methods) > 0, "Should execute at least one method"
        
        # Verify result retrieval
        all_results = scoring.get_all_results()
        assert isinstance(all_results, dict), "get_all_results should return dict"
        assert len(all_results) > 0, "Should have results"
        
        # Verify ranking comparison
        rankings_comparison = scoring.compare_rankings()
        assert rankings_comparison is not None, "compare_rankings should have return value"
        
        # Verify score comparison
        scores_comparison = scoring.compare_scores()
        assert scores_comparison is not None, "compare_scores should have return value"
        
        # Verify result validity
        n_alternatives = data['decision_matrix'].shape[0]
        for method_name, result in all_results.items():
            if isinstance(result, dict):
                if 'ranking' in result:
                    ranking = result['ranking']
                    assert isinstance(ranking, np.ndarray), f"{method_name} ranking should be numpy array"
                    assert TestHelper.is_valid_ranking(ranking, n_alternatives), f"{method_name} ranking should be valid"
                
                if 'scores' in result:
                    scores = result['scores']
                    assert isinstance(scores, np.ndarray), f"{method_name} scores should be numpy array"
                    assert TestHelper.is_valid_scores(scores), f"{method_name} scores should be valid"
        
        print(f"✅ ScoringDecision test passed - executed {len(executed_methods)} methods")
    
    @pytest.mark.smoke
    def test_pairwise_decision_smoke(self):
        """PairwiseDecision functional smoke test"""
        # Use test data
        data = TestData.PAIRWISE_DATA
        thresholds = data['thresholds']
        
        # Create PairwiseDecision instance
        pairwise = PairwiseDecision()
        
        # Call decide method
        pairwise.decide(
            dataset=data['decision_matrix'],
            weights=data['weights'],
            Q=thresholds['Q'],
            P=thresholds['P'],
            V=thresholds['V'],
            S=thresholds['S'],
            F=thresholds['F'],
            B=thresholds['B']
        )
        
        # Verify basic functionality
        executed_methods = pairwise.get_executed_methods()
        assert len(executed_methods) > 0, "Should execute at least one method"
        
        # Verify result retrieval
        all_results = pairwise.get_all_results()
        assert isinstance(all_results, dict), "get_all_results should return dict"
        assert len(all_results) > 0, "Should have results"
        
        # Verify ranking comparison
        rankings_comparison = pairwise.compare_rankings()
        assert rankings_comparison is not None, "compare_rankings should have return value"
        
        # Verify result validity
        n_alternatives = data['decision_matrix'].shape[0]
        for method_name, result in all_results.items():
            if isinstance(result, dict) and 'ranking' in result:
                ranking = result['ranking']
                assert isinstance(ranking, np.ndarray), f"{method_name} ranking should be numpy array"
                assert TestHelper.is_valid_ranking(ranking, n_alternatives), f"{method_name} ranking should be valid"
        
        print(f"✅ PairwiseDecision test passed - executed {len(executed_methods)} methods")
    
    @pytest.mark.smoke
    def test_fuzzy_sub_weighting_smoke(self):
        """FuzzySubWeighting functional smoke test"""
        # Use test data
        data = TestData.FUZZY_SUB_WEIGHTING_DATA
        
        # Create FuzzySubWeighting instance
        fuzzy_sub = FuzzySubWeighting()
        
        # Call decide method
        fuzzy_sub.decide(
            dataset=data['fuzzy_ahp_matrix'],
            mic=data['fuzzy_bwm_data']['mic'],
            lic=data['fuzzy_bwm_data']['lic'],
            criteria_rank=data['fuzzy_fucom_data']['criteria_rank'],
            criteria_priority=data['fuzzy_fucom_data']['criteria_priority']
        )
        
        # Verify basic functionality
        executed_methods = fuzzy_sub.get_executed_methods()
        assert len(executed_methods) > 0, "Should execute at least one method"
        
        # Verify result retrieval
        all_results = fuzzy_sub.get_all_results()
        assert isinstance(all_results, dict), "get_all_results should return dict"
        assert len(all_results) > 0, "Should have results"
        
        # Verify weight comparison
        weights_comparison = fuzzy_sub.compare_weights()
        assert weights_comparison is not None, "compare_weights should have return value"
        
        # Verify fuzzy weight results
        for method_name, result in all_results.items():
            if isinstance(result, dict):
                assert 'weights' in result, f"{method_name} should have weight results"
                weights = result['weights']
                assert isinstance(weights, (list, np.ndarray)), f"{method_name} weights should be list or array"
        
        print(f"✅ FuzzySubWeighting test passed - executed {len(executed_methods)} methods")
    
    @pytest.mark.smoke
    def test_fuzzy_obj_weighting_smoke(self):
        """FuzzyObjWeighting functional smoke test"""
        # Use test data
        data = TestData.FUZZY_OBJ_WEIGHTING_DATA
        
        # Create FuzzyObjWeighting instance
        fuzzy_obj = FuzzyObjWeighting()
        
        # Call decide method
        fuzzy_obj.decide(
            dataset=data['fuzzy_decision_matrix'],
            criterion_type=data['criterion_types']
        )
        
        # Verify basic functionality
        executed_methods = fuzzy_obj.get_executed_methods()
        assert len(executed_methods) > 0, "Should execute at least one method"
        
        # Verify result retrieval
        all_results = fuzzy_obj.get_all_results()
        assert isinstance(all_results, dict), "get_all_results should return dict"
        assert len(all_results) > 0, "Should have results"
        
        # Verify weight comparison
        weights_comparison = fuzzy_obj.compare_weights()
        assert weights_comparison is not None, "compare_weights should have return value"
        
        # Verify fuzzy weight results
        for method_name, result in all_results.items():
            if isinstance(result, dict) and 'weights' in result:
                weights = result['weights']
                assert isinstance(weights, np.ndarray), f"{method_name} weights should be numpy array"
        
        print(f"✅ FuzzyObjWeighting test passed - executed {len(executed_methods)} methods")
    
    @pytest.mark.smoke
    def test_fuzzy_scoring_decision_smoke(self):
        """FuzzyScoringDecision functional smoke test"""
        # Use test data
        data = TestData.FUZZY_SCORING_DATA
        
        # Create FuzzyScoringDecision instance
        fuzzy_scoring = FuzzyScoringDecision()
        
        # Call decide method
        fuzzy_scoring.decide(
            dataset=data['fuzzy_decision_matrix'],
            weights=data['fuzzy_weights'],
            criterion_type=data['criterion_types']
        )
        
        # Verify basic functionality
        executed_methods = fuzzy_scoring.get_executed_methods()
        assert len(executed_methods) > 0, "Should execute at least one method"
        
        # Verify result retrieval
        all_results = fuzzy_scoring.get_all_results()
        assert isinstance(all_results, dict), "get_all_results should return dict"
        assert len(all_results) > 0, "Should have results"
        
        # Verify ranking comparison
        rankings_comparison = fuzzy_scoring.compare_rankings()
        assert rankings_comparison is not None, "compare_rankings should have return value"
        
        # Verify score comparison
        scores_comparison = fuzzy_scoring.compare_scores()
        assert scores_comparison is not None, "compare_scores should have return value"
        
        # Verify fuzzy decision results
        n_alternatives = data['fuzzy_decision_matrix'].shape[0]
        for method_name, result in all_results.items():
            if isinstance(result, dict):
                if 'ranking' in result:
                    ranking = result['ranking']
                    assert isinstance(ranking, np.ndarray), f"{method_name} ranking should be numpy array"
                    assert TestHelper.is_valid_ranking(ranking, n_alternatives), f"{method_name} ranking should be valid"
        
        print(f"✅ FuzzyScoringDecision test passed - executed {len(executed_methods)} methods")
    
    @pytest.mark.smoke
    def test_all_methods_coverage(self):
        """Test all methods coverage - ensure all public methods are called"""
        decision_classes = [
            SubWeighting, ObjWeighting, ScoringDecision, PairwiseDecision,
            FuzzySubWeighting, FuzzyObjWeighting, FuzzyScoringDecision
        ]
        
        method_coverage = {}
        
        for decision_class in decision_classes:
            class_name = decision_class.__name__
            
            # Create instance
            instance = decision_class()
            
            # Get all public methods
            public_methods = [method for method in dir(instance) 
                            if not method.startswith('_') and callable(getattr(instance, method))]
            
            # Check if key methods exist
            required_methods = ['decide', 'get_all_results']
            optional_methods = ['compare_weights', 'compare_rankings', 'compare_scores']
            
            coverage = {
                'total_public_methods': len(public_methods),
                'has_decide': 'decide' in public_methods,
                'has_get_all_results': 'get_all_results' in public_methods,
                'has_compare_methods': sum(1 for method in optional_methods if method in public_methods),
                'all_required_present': all(method in public_methods for method in required_methods)
            }
            
            method_coverage[class_name] = coverage
            
            # Verify required methods exist
            assert coverage['has_decide'], f"{class_name} should have decide method"
            assert coverage['has_get_all_results'], f"{class_name} should have get_all_results method"
            assert coverage['all_required_present'], f"{class_name} missing required methods"
        
        print(f"✅ Method coverage test passed - all {len(decision_classes)} decision classes have required methods")
        
        # Print coverage details
        for class_name, coverage in method_coverage.items():
            print(f"  - {class_name}: {coverage['total_public_methods']} public methods, "
                  f"{coverage['has_compare_methods']} compare methods")
    
    @pytest.mark.smoke
    @pytest.mark.performance
    def test_performance_baseline(self):
        """Performance baseline test - measure basic execution time of all decision classes"""
        test_data_map = {
            SubWeighting: {
                'dataset': TestData.SUB_WEIGHTING_DATA['ahp_matrix'],
                'mic': TestData.SUB_WEIGHTING_DATA['bwm_data']['mic'],
                'lic': TestData.SUB_WEIGHTING_DATA['bwm_data']['lic'],
                'criteria_rank': TestData.SUB_WEIGHTING_DATA['fucom_data']['criteria_rank'],
                'criteria_priority': TestData.SUB_WEIGHTING_DATA['fucom_data']['criteria_priority']
            },
            ObjWeighting: {
                'dataset': TestData.OBJ_WEIGHTING_DATA['decision_matrix'],
                'criterion_type': TestData.OBJ_WEIGHTING_DATA['criterion_types']
            },
            ScoringDecision: {
                'dataset': TestData.SCORING_DATA['decision_matrix'],
                'criterion_type': TestData.SCORING_DATA['criterion_types'],
                'weights': TestData.SCORING_DATA['weights']
            },
            PairwiseDecision: {
                'dataset': TestData.PAIRWISE_DATA['decision_matrix'],
                'weights': TestData.PAIRWISE_DATA['weights'],
                'Q': TestData.PAIRWISE_DATA['thresholds']['Q'],
                'P': TestData.PAIRWISE_DATA['thresholds']['P'],
                'V': TestData.PAIRWISE_DATA['thresholds']['V'],
                'S': TestData.PAIRWISE_DATA['thresholds']['S'],
                'F': TestData.PAIRWISE_DATA['thresholds']['F']
            },
            FuzzySubWeighting: {
                'dataset': TestData.FUZZY_SUB_WEIGHTING_DATA['fuzzy_ahp_matrix'],
                'mic': TestData.FUZZY_SUB_WEIGHTING_DATA['fuzzy_bwm_data']['mic'],
                'lic': TestData.FUZZY_SUB_WEIGHTING_DATA['fuzzy_bwm_data']['lic']
            },
            FuzzyObjWeighting: {
                'dataset': TestData.FUZZY_OBJ_WEIGHTING_DATA['fuzzy_decision_matrix'],
                'criterion_type': TestData.FUZZY_OBJ_WEIGHTING_DATA['criterion_types']
            },
            FuzzyScoringDecision: {
                'dataset': TestData.FUZZY_SCORING_DATA['fuzzy_decision_matrix'],
                'weights': TestData.FUZZY_SCORING_DATA['fuzzy_weights'],
                'criterion_type': TestData.FUZZY_SCORING_DATA['criterion_types']
            }
        }
        
        execution_times = {}
        
        for decision_class, test_data in test_data_map.items():
            class_name = decision_class.__name__
            
            # Measure execution time
            instance = decision_class()
            _, execution_time = TestHelper.measure_execution_time(
                instance.decide, **test_data
            )
            
            execution_times[class_name] = execution_time
        
        # Generate performance report
        performance_report = TestHelper.compare_execution_times(execution_times, threshold=2.0)
        
        print(f"✅ Performance baseline test completed")
        for class_name, time_taken in execution_times.items():
            performance = performance_report[class_name]
            print(f"  - {class_name}: {time_taken:.3f} seconds ({performance})")
        
        # Performance assertions - all methods should complete within reasonable time
        # Fuzzy methods usually take longer
        for class_name, time_taken in execution_times.items():
            if 'Fuzzy' in class_name:
                threshold = 60.0  # Allow more time for fuzzy methods
            else:
                threshold = 10.0  # Regular methods
            assert time_taken < threshold, f"{class_name} execution time too long: {time_taken:.3f} seconds"


if __name__ == "__main__":
    # Quick test when running directly
    import sys
    sys.path.append('..')
    
    test_instance = TestSmokeTests()
    test_instance.setup_method()
    
    try:
        test_instance.test_sub_weighting_smoke()
        test_instance.test_obj_weighting_smoke()
        test_instance.test_scoring_decision_smoke()
        test_instance.test_pairwise_decision_smoke()
        test_instance.test_fuzzy_sub_weighting_smoke()
        test_instance.test_fuzzy_obj_weighting_smoke()
        test_instance.test_fuzzy_scoring_decision_smoke()
        test_instance.test_all_methods_coverage()
        test_instance.test_performance_baseline()
        print("\n🎉 All smoke tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()