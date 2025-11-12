"""
pyDecision 包装器的方法注册表。

本模块维护一个所有可用决策方法的综合注册表，
包括其必需参数、可选参数和输出类型。
"""

from typing import Dict, List, Any, Optional
from enum import Enum


class OutputType(Enum):
    """方法输出类型的枚举。"""
    WEIGHTS = "weights"
    SCORES = "scores"
    RANKINGS = "rankings"
    MULTIPLE_SCORES = "multiple_scores"
    RELATIONAL = "relational"
    CLASSIFICATION = "classification"
    COMPREHENSIVE = "comprehensive"


class MethodCategory(Enum):
    """方法类别的枚举。"""
    SUBJECTIVE_WEIGHTING = "subjective_weighting"
    OBJECTIVE_WEIGHTING = "objective_weighting"
    SCORING = "scoring"
    PAIRWISE = "pairwise"
    FUZZY = "fuzzy"
    CLASSIFICATION = "classification"


class MethodRegistry:
    """
    所有决策方法的中央注册表。
    
    该类维护每个方法的元数据，包括：
    - 必需和可选参数
    - 输出类型
    - 类别
    - pyDecision 函数映射
    """
    
    # 权重方法注册表
    SUB_WEIGHTING_METHODS = {
        # 主观权重方法
        'ahp': {
            'category': MethodCategory.SUBJECTIVE_WEIGHTING,
            'required': ['dataset'],  # 两两比较矩阵
            'optional': ['wd'],  # 权重推导方法
            'output_type': OutputType.WEIGHTS,
            'description': 'Analytic Hierarchy Process'
        },
        'bwm': {
            'category': MethodCategory.SUBJECTIVE_WEIGHTING,
            'required': ['mic', 'lic'],  # 最佳到其他和其他到最差向量
            'optional': ['eps_penalty', 'verbose'],
            'output_type': OutputType.WEIGHTS,
            'description': 'Best-Worst Method'
        },
        'simplified_bwm': {
            'category': MethodCategory.SUBJECTIVE_WEIGHTING,
            'required': ['mic', 'lic'],
            'optional': ['alpha', 'verbose'],
            'output_type': OutputType.WEIGHTS,
            'description': 'Simplified Best-Worst Method'
        },
        'fucom': {
            'category': MethodCategory.SUBJECTIVE_WEIGHTING,
            'required': ['criteria_rank', 'criteria_priority'],
            'optional': ['sort_criteria', 'verbose'],
            'output_type': OutputType.WEIGHTS,
            'description': 'Full Consistency Method'
        },
        'roc': {
            'category': MethodCategory.SUBJECTIVE_WEIGHTING,
            'required': ['criteria_rank'],
            'optional': [],
            'output_type': OutputType.WEIGHTS,
            'description': 'Rank Order Centroid'
        },
        'rrw': {
            'category': MethodCategory.SUBJECTIVE_WEIGHTING,
            'required': ['criteria_rank'],
            'optional': [],
            'output_type': OutputType.WEIGHTS,
            'description': 'Rank Reciprocal Weighting'
        },
        'rsw': {
            'category': MethodCategory.SUBJECTIVE_WEIGHTING,
            'required': ['criteria_rank'],
            'optional': [],
            'output_type': OutputType.WEIGHTS,
            'description': 'Rank Sum Weights'
        },
        'dematel': {
            'category': MethodCategory.SUBJECTIVE_WEIGHTING,
            'required': ['dataset'],
            'optional': ['size_x', 'size_y'],
            'output_type': OutputType.WEIGHTS,
            'description': 'Decision Making Trial and Evaluation Laboratory'
        },
        'wings': {
            'category': MethodCategory.SUBJECTIVE_WEIGHTING,
            'required': ['dataset'],
            'optional': ['size_x', 'size_y'],
            'output_type': OutputType.WEIGHTS,
            'description': 'Weighted Influence Non-linear Gauge System'
        }
    } 
    
    OBJ_WEIGHTING_METHODS = {
        # 客观权重方法
        'critic': {
            'category': MethodCategory.OBJECTIVE_WEIGHTING,
            'required': ['dataset', 'criterion_type'],
            'optional': [],
            'output_type': OutputType.WEIGHTS,
            'description': 'CRiteria Importance Through Intercriteria Correlation'
        },
        'entropy': {
            'category': MethodCategory.OBJECTIVE_WEIGHTING,
            'required': ['dataset', 'criterion_type'],
            'optional': [],
            'output_type': OutputType.WEIGHTS,
            'description': 'Entropy Weight Method'
        },
        'idocriw': {
            'category': MethodCategory.OBJECTIVE_WEIGHTING,
            'required': ['dataset', 'criterion_type'],
            'optional': ['verbose'],
            'output_type': OutputType.WEIGHTS,
            'description': 'Integrated Determination of Objective CRIteria Weights'
        },
        'merec': {
            'category': MethodCategory.OBJECTIVE_WEIGHTING,
            'required': ['dataset', 'criterion_type'],
            'optional': [],
            'output_type': OutputType.WEIGHTS,
            'description': 'MEthod based on the Removal Effects of Criteria'
        },
        'mpsi': {
            'category': MethodCategory.OBJECTIVE_WEIGHTING,
            'required': ['dataset', 'criterion_type'],
            'optional': [],
            'output_type': OutputType.WEIGHTS,
            'description': 'Modified Preference Selection Index'
        },
        'seca': {
            'category': MethodCategory.OBJECTIVE_WEIGHTING,
            'required': ['dataset', 'criterion_type'],
            'optional': ['beta'],
            'output_type': OutputType.WEIGHTS,
            'description': 'Simultaneous Evaluation of Criteria and Alternatives'
        },
        'cilos': {
            'category': MethodCategory.OBJECTIVE_WEIGHTING,
            'required': ['dataset', 'criterion_type'],
            'optional': [],
            'output_type': OutputType.WEIGHTS,
            'description': 'Criterion Impact LOSs'
        }
    }
    
    # 评分方法注册表
    SCORING_METHODS = {
        'topsis': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Technique for Order of Preference by Similarity to Ideal Solution'
        },
        'vikor': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['strategy_coefficient', 'graph', 'verbose'],
            'output_type': OutputType.MULTIPLE_SCORES,
            'description': 'VIseKriterijumska Optimizacija I Kompromisno Resenje'
        },
        'aras': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Additive Ratio ASsessment'
        },
        'cocoso': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['L', 'graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'COmbined COmpromise SOlution'
        },
        'codas': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['lmbd', 'graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'COmbinative Distance-based ASsessment'
        },
        'copras': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'COmplex PRoportional ASsessment'
        },
        'cradis': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Compromise Ranking of Alternatives from Distance to Ideal Solution'
        },
        'edas': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Evaluation based on Distance from Average Solution'
        },
        'lmaw': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Logarithmic Methodology of Additive Weights'
        },
        'mabac': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Multi-Attributive Border Approximation area Comparison'
        },
        'mairca': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Multi-Attributive Ideal-Real Comparative Analysis'
        },
        'marcos': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Measurement of Alternatives and Ranking according to COmpromise Solution'
        },
        'maut': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type', 'utility_functions'],
            'optional': ['step_size', 'graph', 'verbose'],
            'output_type': OutputType.COMPREHENSIVE,
            'description': 'Multi-Attribute Utility Theory'
        },
        'moora': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Multi-Objective Optimization on the basis of Ratio Analysis'
        },
        'moosra': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Multi-Objective Optimization on the basis of Simple Ratio Analysis'
        },
        'multimoora': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'criterion_type'],
            'optional': ['graph'],
            'output_type': OutputType.MULTIPLE_SCORES,
            'description': 'MOORA plus Full Multiplicative Form'
        },
        'rafsi': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['ideal', 'anti_ideal', 'n_i', 'n_k', 'graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Ranking of Alternatives through Functional mapping of criterion sub-intervals into a Single Interval'
        },
        'saw': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Simple Additive Weighting'
        },
        'smart': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'grades', 'lower', 'upper', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Simple Multi-Attribute Rating Technique'
        },
        'spotis': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type', 's_min', 's_max'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Stable Preference Ordering Towards Ideal Solution'
        },
        'todim': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['teta', 'graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'TOmada de Decisao Interativa Multicriterio'
        },
        'waspas': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type', 'lambda_value'],
            'optional': ['graph'],
            'output_type': OutputType.MULTIPLE_SCORES,
            'description': 'Weighted Aggregated Sum Product ASsessment'
        },
        'wisp': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['simplified', 'graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Weighted Sum-Product'
        },
        'gra': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['epsilon', 'graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Grey Relational Analysis'
        },
        'macbeth': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Measuring Attractiveness by a Categorical Based Evaluation TecHnique'
        },
        'mara': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Magnitude of the Area for the Ranking of Alternatives'
        },
        'ocra': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Operational Competitiveness RAting'
        },
        'oreste': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['alpha', 'graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Organisation, Rangement Et Synthese de donnes relaTionnEles'
        },
        'piv': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Proximity Indexed Value'
        },
        'rov': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Range Of Values'
        },
        'borda': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Borda Count Method'
        },
        'copeland': {
            'category': MethodCategory.SCORING,
            'required': ['dataset', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Copeland Method'
        },
        'psi': {
            'category': MethodCategory.OBJECTIVE_WEIGHTING,
            'required': ['dataset', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.WEIGHTS,
            'description': 'Preference Selection Index'
        }
    }
    
    # 两两比较方法注册表
    PAIRWISE_METHODS = {
        'electre_iii': {
            'category': MethodCategory.PAIRWISE,
            'required': ['dataset', 'P', 'Q', 'V', 'W'],
            'optional': ['graph'],
            'output_type': OutputType.RANKINGS,
            'description': 'ELECTRE III'
        },
        'electre_iv': {
            'category': MethodCategory.PAIRWISE,
            'required': ['dataset', 'P', 'Q', 'V'],
            'optional': ['graph'],
            'output_type': OutputType.RANKINGS,
            'description': 'ELECTRE IV'
        },
        'promethee_ii': {
            'category': MethodCategory.PAIRWISE,
            'required': ['dataset', 'W', 'Q', 'S', 'P', 'F'],
            'optional': ['sort', 'topn'],
            'output_type': OutputType.SCORES,
            'description': 'PROMETHEE II'
        }
    }
    
    # 模糊主观权重方法注册表
    FUZZY_SUB_WEIGHTING_METHODS = {
        'fuzzy_ahp': {
            'category': MethodCategory.FUZZY,
            'required': ['dataset'],  # 模糊两两比较矩阵
            'optional': [],
            'output_type': OutputType.WEIGHTS,
            'description': 'Fuzzy Analytic Hierarchy Process'
        },
        'fuzzy_bwm': {
            'category': MethodCategory.FUZZY,
            'required': ['mic', 'lic'],
            'optional': ['eps_penalty', 'verbose'],
            'output_type': OutputType.WEIGHTS,
            'description': 'Fuzzy Best-Worst Method'
        },
        'fuzzy_fucom': {
            'category': MethodCategory.FUZZY,
            'required': ['criteria_rank', 'criteria_priority'],
            'optional': ['n_starts', 'sort_criteria', 'verbose'],
            'output_type': OutputType.WEIGHTS,
            'description': 'Fuzzy FUCOM'
        }
    }
    
    # 模糊客观权重方法注册表
    FUZZY_OBJ_WEIGHTING_METHODS = {
        'fuzzy_critic': {
            'category': MethodCategory.FUZZY,
            'required': ['dataset', 'criterion_type'],
            'optional': [],
            'output_type': OutputType.WEIGHTS,
            'description': 'Fuzzy CRITIC'
        },
        'fuzzy_merec': {
            'category': MethodCategory.FUZZY,
            'required': ['dataset', 'criterion_type'],
            'optional': [],
            'output_type': OutputType.WEIGHTS,
            'description': 'Fuzzy MEREC'
        }
    }
    
    # 模糊评分方法注册表
    FUZZY_SCORING_METHODS = {
        'fuzzy_aras': {
            'category': MethodCategory.FUZZY,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Fuzzy ARAS'
        },
        'fuzzy_copras': {
            'category': MethodCategory.FUZZY,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Fuzzy COPRAS'
        },
        'fuzzy_edas': {
            'category': MethodCategory.FUZZY,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Fuzzy EDAS'
        },
        'fuzzy_moora': {
            'category': MethodCategory.FUZZY,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Fuzzy MOORA'
        },
        'fuzzy_ocra': {
            'category': MethodCategory.FUZZY,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Fuzzy OCRA'
        },
        'fuzzy_topsis': {
            'category': MethodCategory.FUZZY,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph', 'verbose'],
            'output_type': OutputType.SCORES,
            'description': 'Fuzzy TOPSIS'
        },
        'fuzzy_vikor': {
            'category': MethodCategory.FUZZY,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['strategy_coefficient', 'graph', 'verbose'],
            'output_type': OutputType.MULTIPLE_SCORES,
            'description': 'Fuzzy VIKOR'
        },
        'fuzzy_waspas': {
            'category': MethodCategory.FUZZY,
            'required': ['dataset', 'weights', 'criterion_type'],
            'optional': ['graph'],
            'output_type': OutputType.SCORES,
            'description': 'Fuzzy WASPAS'
        }
    }
    
    # 分类方法注册表
    CLASSIFICATION_METHODS = {
        'electre_tri_b': {
            'category': MethodCategory.CLASSIFICATION,
            'required': ['dataset', 'P', 'Q', 'V', 'W', 'B'],
            'optional': ['cut_level', 'rule', 'verbose'],
            'output_type': OutputType.CLASSIFICATION,
            'description': 'ELECTRE TRI-B'
        },
        'cpp_tri': {
            'category': MethodCategory.CLASSIFICATION,
            'required': ['decision_matrix'],
            'optional': ['weights', 'profiles', 'num_cat', 'indep_criteria', 'rule', 'verbose'],
            'output_type': OutputType.CLASSIFICATION,
            'description': 'Comparison with Profiles using Thresholds'
        }
    }
    
    @classmethod
    def get_all_methods(cls) -> Dict[str, Dict[str, Any]]:
        """获取所有方法的注册表。"""
        all_methods = {}
        all_methods.update(cls.OBJ_WEIGHTING_METHODS)
        all_methods.update(cls.SUB_WEIGHTING_METHODS)
        all_methods.update(cls.SCORING_METHODS)
        all_methods.update(cls.PAIRWISE_METHODS)
        all_methods.update(cls.CLASSIFICATION_METHODS)
        all_methods.update(cls.FUZZY_SUB_WEIGHTING_METHODS)
        all_methods.update(cls.FUZZY_OBJ_WEIGHTING_METHODS)
        all_methods.update(cls.FUZZY_SCORING_METHODS)
        all_methods.update(cls.SPECIAL_METHODS)
        return all_methods
    
    @classmethod
    def get_methods_by_category(cls, category: MethodCategory) -> Dict[str, Dict[str, Any]]:
        """获取特定类别中的所有方法。"""
        all_methods = cls.get_all_methods()
        return {
            name: info 
            for name, info in all_methods.items() 
            if info['category'] == category
        }
    
    @classmethod
    def get_methods_by_output_type(cls, output_type: OutputType) -> Dict[str, Dict[str, Any]]:
        """获取具有特定输出类型的所有方法。"""
        all_methods = cls.get_all_methods()
        return {
            name: info 
            for name, info in all_methods.items() 
            if info['output_type'] == output_type
        }
    
    @classmethod
    def get_method_info(cls, method_name: str) -> Optional[Dict[str, Any]]:
        """获取有关特定方法的信息。"""
        all_methods = cls.get_all_methods()
        return all_methods.get(method_name)
    
    @classmethod
    def get_required_parameters(cls, method_name: str) -> Optional[List[str]]:
        """获取特定方法的必需参数。"""
        method_info = cls.get_method_info(method_name)
        return method_info['required'] if method_info else None
    
    @classmethod
    def get_optional_parameters(cls, method_name: str) -> Optional[List[str]]:
        """获取特定方法的可选参数。"""
        method_info = cls.get_method_info(method_name)
        return method_info['optional'] if method_info else None
    
    @classmethod
    def validate_parameters(cls, method_name: str, **kwargs) -> tuple[bool, List[str]]:
        """
        验证提供的参数是否足以满足方法的要求。
        
        返回:
            一个元组 (is_valid, missing_parameters)
        """
        method_info = cls.get_method_info(method_name)
        if not method_info:
            return False, [f"Unknown method: {method_name}"]
        
        required = set(method_info['required'])
        provided = set(k for k, v in kwargs.items() if v is not None)
        missing = list(required - provided)
        
        return len(missing) == 0, missing
