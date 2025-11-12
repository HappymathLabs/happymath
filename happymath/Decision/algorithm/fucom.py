###############################################################################

# Required Libraries
import numpy as np
import warnings
warnings.filterwarnings('ignore', message = 'delta_grad == 0.0. Check if the approximated')
warnings.filterwarnings('ignore', message = 'Values in x were outside bounds during a minimize step, clipping to bounds')

from scipy.optimize import minimize, Bounds, LinearConstraint

###############################################################################

# Function: FUCOM (Full Consistency Method)
def fucom(criteria_rank, criteria_priority, sort_criteria = True, verbose = True):
    
    ################################################
    
    def target_function(variables):
        variables       = np.array(variables)
        
        # Deviation from condition 1: w_k / w_{k+1} = phi_{k/(k+1)}
        ratios_1        = variables[:-1] / variables[1:]
        target_ratios_1 = np.array(criteria_priority)
        chi_1           = np.abs(ratios_1 - target_ratios_1)
        
        # Deviation from condition 2: w_k / w_{k+2} = phi_{k/(k+1)} * phi_{(k+1)/(k+2)}
        ratios_2        = variables[:-2] / variables[2:]
        target_ratios_2 = np.array(criteria_priority[:-1]) * np.array(criteria_priority[1:])
        chi_2           = np.abs(ratios_2 - target_ratios_2)

        chi             = np.hstack((chi_1, chi_2))
        return np.max(chi)
    
    ################################################
    
    n_criteria = len(criteria_rank)
    np.random.seed(42)
    variables   = np.random.uniform(low = 0.001, high = 1.0, size = n_criteria)
    variables   = variables / np.sum(variables)
    bounds      = Bounds(0.0001, 1.0)
    constraints = LinearConstraint(np.ones(n_criteria), 1, 1)
    results     = minimize(target_function, variables, method = 'SLSQP', constraints = constraints, bounds = bounds)
    weights     = results.x
    if (sort_criteria == True):
        
        sorted_indices = np.argsort(criteria_rank)
        
        final_weights = np.zeros(n_criteria)
        final_weights[sorted_indices] = weights
        weights = final_weights
    if (verbose == True):
        print('Chi:', np.round(results.fun, 4))
    return weights

###############################################################################
