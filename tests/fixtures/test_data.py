"""
Test data module

Contains various standard datasets for testing, referencing sample data in test_all_decision.ipynb
"""
import numpy as np


class TestData:
    """Test data collection class"""
    
    # SubWeighting test data
    SUB_WEIGHTING_DATA = {
        'ahp_matrix': np.array([
            [1, 3, 5, 7],
            [1/3, 1, 3, 5],
            [1/5, 1/3, 1, 3],
            [1/7, 1/5, 1/3, 1]
        ]),
        'bwm_data': {
            'mic': np.array([1, 3, 5, 7]),  # Best criterion to other criteria
            'lic': np.array([7, 5, 3, 1])   # Other criteria to worst criterion
        },
        'fucom_data': {
            'criteria_rank': np.array([1, 4, 3, 2]),  # Criteria ranking
            'criteria_priority': np.array([1.2, 1.1, 1.2])  # Adjacent criteria priority ratio
        }
    }
    
    # ObjWeighting test data
    OBJ_WEIGHTING_DATA = {
        'decision_matrix': np.array([
            [10, 20, 5,  80],
            [12, 18, 6,  85],
            [11, 22, 5.5,78],
            [15, 15, 7,  90],
            [9,  25, 4.5,75]
        ]),
        'criterion_types': ['min', 'max', 'min', 'max']
    }
    
    # ScoringDecision test data
    SCORING_DATA = {
        'decision_matrix': np.array([
            [250, 16, 12, 5, 4],
            [200, 16, 8,  3, 3],
            [300, 32, 16, 4, 4],
            [275, 32, 8,  4, 5]
        ]),
        'criterion_types': ['min', 'max', 'max', 'max', 'max'],
        'weights': np.array([0.25, 0.15, 0.2, 0.2, 0.2]),
        'waspas_params': {
            'lambda_value': 0.5
        },
        'spotis_params': {
            's_min': np.array([180, 15, 7, 2, 2]),
            's_max': np.array([320, 35, 18, 6, 6])
        },
        'smart_params': {
            'grades': np.array([0.25, 0.15, 0.2, 0.2, 0.2]),
            'lower': np.array([180, 15, 7, 2, 2]),
            'upper': np.array([320, 35, 18, 6, 6]),
            'utility_functions': ['exp', 'step', 'quad', 'log', 'ln']
        }
    }
    
    # PairwiseDecision test data
    PAIRWISE_DATA = {
        'decision_matrix': np.array([
            [5, 85, 70, 15, 0.8],
            [4, 92, 65, 12, 0.9],
            [5, 80, 85, 20, 0.7],
            [6, 75, 80, 18, 0.85]
        ]),
        'weights': np.array([0.25, 0.30, 0.20, 0.15, 0.10]),
        'thresholds': {
            'Q': np.array([1, 5, 5, 2, 0.05]),     # Indifference threshold
            'P': np.array([2, 15, 15, 5, 0.1]),    # Preference threshold
            'V': np.array([4, 20, 20, 8, 0.15]),   # Veto threshold
            'S': np.array([1, 10, 10, 3, 0.08]),   # Strict preference threshold
            'F': np.array([5, 5, 5, 5, 5]),        # Preference function type
            'B': np.array([3, 70, 60, 10, 0.6])    # Category boundary
        }
    }
    
    # FuzzySubWeighting test data
    FUZZY_SUB_WEIGHTING_DATA = {
        'fuzzy_ahp_matrix': np.array([
            [[1, 1, 1], [2, 3, 4], [3, 4, 5]],
            [[1/4, 1/3, 1/2], [1, 1, 1], [1, 2, 3]],
            [[1/5, 1/4, 1/3], [1/3, 1/2, 1], [1, 1, 1]]
        ]),
        'fuzzy_bwm_data': {
            'mic': np.array([
                (1, 1, 1),        # C1/C1
                (2.5, 3, 3.5),    # C1/C2
                (3.5, 4, 4.5)     # C1/C3
            ], dtype=object),
            'lic': np.array([
                (3.5, 4, 4.5),    # C1/C3
                (2.5, 3, 3.5),    # C2/C3
                (1, 1, 1)         # C3/C3
            ], dtype=object)
        },
        'fuzzy_fucom_data': {
            'criteria_rank': np.array([1, 2, 3]),
            'criteria_priority': np.array([
                (1.5, 2.0, 2.5),
                (2.0, 2.5, 3.0)
            ], dtype=object)
        }
    }
    
    # FuzzyObjWeighting test data
    FUZZY_OBJ_WEIGHTING_DATA = {
        'fuzzy_decision_matrix': np.array([
            [[6, 7, 8], [2, 3, 4], [8, 9, 9]],
            [[7, 8, 9], [3, 4, 5], [7, 8, 9]],
            [[8, 9, 9], [1, 2, 3], [6, 7, 8]],
            [[7, 8, 9], [4, 5, 6], [7, 8, 8]]
        ]),
        'criterion_types': ['max', 'min', 'max']
    }
    
    # FuzzyScoringDecision test data
    FUZZY_SCORING_DATA = {
        'fuzzy_decision_matrix': np.array([
            [[5, 6, 7], [3, 4, 5], [4, 5, 6], [1, 2, 3]],
            [[7, 8, 9], [6, 7, 8], [5, 6, 7], [4, 5, 6]],
            [[3, 4, 5], [4, 5, 6], [6, 7, 8], [5, 6, 7]]
        ]),
        'fuzzy_weights': np.array([
            (0.3038148036677145, 0.3603762029296692, 0.4647795493123453),
            (0.2587679917494022, 0.3151640292910815, 0.4086518556905072),
            (0.14483084523706044, 0.21023162093669034, 0.2855020836159773),
            (0.078181245602842, 0.09845102772413762, 0.11858010159783688)
        ]),
        'criterion_types': ['min', 'min', 'min', 'min']
    }


class BoundaryTestData:
    """Boundary test data"""
    
    # Minimum scale test data
    MIN_DATA = {
        'matrix_2x2': np.array([[1, 2], [3, 4]]),
        'weights_2': np.array([0.3, 0.7]),
        'criteria_types_2': ['min', 'max']
    }
    
    # Large scale test data  
    LARGE_DATA = {
        'matrix_100x20': None,  # Will be generated at runtime
        'weights_20': None,     # Will be generated at runtime
        'criteria_types_20': None  # Will be generated at runtime
    }
    
    # Special values test data
    SPECIAL_VALUES = {
        'zero_matrix': np.array([[0, 0, 0], [0, 0, 0]]),
        'ones_matrix': np.array([[1, 1, 1], [1, 1, 1]]),
        'identical_matrix': np.array([[5, 5, 5], [5, 5, 5]]),
        'extreme_small': np.array([[1e-10, 1e-9], [1e-8, 1e-7]]),
        'extreme_large': np.array([[1e10, 1e9], [1e8, 1e7]])
    }
    
    @classmethod
    def generate_large_data(cls, n_alternatives=100, n_criteria=20):
        """Generate large scale test data"""
        np.random.seed(42)
        cls.LARGE_DATA['matrix_100x20'] = np.random.rand(n_alternatives, n_criteria) * 100
        cls.LARGE_DATA['weights_20'] = np.random.dirichlet(np.ones(n_criteria))
        cls.LARGE_DATA['criteria_types_20'] = np.random.choice(['min', 'max'], n_criteria).tolist()


class RobustnessTestData:
    """Robustness test data"""
    
    # Invalid input data
    INVALID_INPUTS = {
        'none_values': {
            'matrix': None,
            'weights': None,
            'criterion_type': None
        },
        'empty_arrays': {
            'matrix': np.array([]),
            'weights': np.array([]),
            'criterion_type': []
        },
        'wrong_types': {
            'matrix': "not_a_matrix",
            'weights': "not_weights", 
            'criterion_type': "not_types"
        },
        'dimension_mismatch': {
            'matrix': np.array([[1, 2, 3], [4, 5, 6]]),
            'weights': np.array([0.5, 0.5]),  # Should be 3 weights
            'criterion_type': ['min', 'max']  # Should be 3 types
        }
    }
    
    # Numerical issues data
    NUMERICAL_ISSUES = {
        'nan_matrix': np.array([[np.nan, 1, 2], [3, 4, 5]]),
        'inf_matrix': np.array([[np.inf, 1, 2], [3, 4, 5]]),
        'negative_inf': np.array([[-np.inf, 1, 2], [3, 4, 5]]),
        'negative_values': np.array([[-1, -2, -3], [-4, -5, -6]]),
        'very_large': np.array([[1e308, 1e307, 1e306], [1, 2, 3]]),
        'very_small': np.array([[1e-308, 1e-307, 1e-306], [1, 2, 3]])
    }
    
    # Weight related issues
    WEIGHT_ISSUES = {
        'negative_weights': np.array([-0.1, 0.6, 0.5]),
        'zero_weights': np.array([0, 0, 0]),
        'sum_not_one': np.array([0.1, 0.2, 0.3]),  # Sum not equal to 1
        'single_weight_one': np.array([1, 0, 0]),
        'nan_weights': np.array([0.3, np.nan, 0.4]),
        'inf_weights': np.array([0.3, np.inf, 0.4])
    }