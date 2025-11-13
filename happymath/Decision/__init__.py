"""
Decision Analysis Framework
==========================

A comprehensive framework for multi-criteria decision making (MCDM) with intelligent 
method selection and unified interfaces.

Main Components:
- SubWeighting: Subjective weight calculation methods (AHP, BWM, FUCOM, ROC, etc.)
- ObjWeighting: Objective weight calculation methods (CRITIC, Entropy, MEREC, PSI, etc.)
- ScoringDecision: Alternative scoring methods (TOPSIS, VIKOR, SAW, etc.)
- PairwiseDecision: Outranking methods (ELECTRE, PROMETHEE families)
- FuzzyDecision: Fuzzy decision-making methods

Quick Start:
>>> from methods.obj_weighting import ObjWeighting
>>> from methods.scoring import ScoringDecision
>>> import numpy as np
>>>
>>> # Decision matrix and criteria types
>>> dm_data = np.array([[250, 16, 12], [200, 16, 8], [300, 32, 16]])
>>> criteria = ['min', 'max', 'max']
>>>
>>> # Calculate weights and rankings
>>> weighting = ObjWeighting()
>>> weights = weighting.decide(dataset=dm_data, criterion_type=criteria).get_weights()
>>> 
>>> scoring = ScoringDecision()
>>> rankings = scoring.decide(dataset=dm_data, weights=weights, criterion_type=criteria).get_rankings()
>>> print(rankings)

For more information, see the documentation and examples.
"""

from .methods.sub_weighting import SubWeighting
from .methods.obj_weighting import ObjWeighting
from .methods.scoring import ScoringDecision
from .methods.pairwise import PairwiseDecision
from .methods.fuzzy_obj_weighting import FuzzyObjWeighting
from .methods.fuzzy_sub_weighting import FuzzySubWeighting
from .methods.fuzzy_scoring import FuzzyScoringDecision
from .results.result_manager import ResultManager, MethodResult
from .core.method_registry import MethodRegistry, MethodCategory
from .core.validators import ParameterValidator

# Removed hardcoded metadata - use happymath.__version__ instead

__all__ = [
    # Main classes
    'SubWeighting',
    'ObjWeighting',
    'ScoringDecision', 
    'PairwiseDecision',
    'FuzzyObjWeighting',
    'FuzzySubWeighting',
    'FuzzyScoringDecision',
    
    # Result management
    'ResultManager',
    'MethodResult',
    
    # Core components
    'MethodRegistry',
    'MethodCategory',
    'ParameterValidator',
]