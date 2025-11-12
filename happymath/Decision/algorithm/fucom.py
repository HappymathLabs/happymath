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
        # 使用 argsort 对权重进行排序，以匹配原始准则顺序
        # criteria_rank 是一个排名数组，例如 [1, 4, 3, 2, 5]
        # argsort() 会返回原始索引，使得按这些索引排序后，数组会变为有序
        # 例如，np.argsort([1, 4, 3, 2, 5]) -> [0, 3, 2, 1, 4]
        # 这意味着 rank 1 的准则在原始数组的索引是 0
        # rank 2 的准则在原始数组的索引是 3
        
        # `weights` 数组是优化后的结果，其顺序是按照重要性从高到低排列的。
        # 我们需要将 `weights` 的值放回对应的 C1, C2, ... 的位置。
        sorted_indices = np.argsort(criteria_rank)
        
        final_weights = np.zeros(n_criteria)
        final_weights[sorted_indices] = weights
        weights = final_weights
    if (verbose == True):
        print('Chi:', np.round(results.fun, 4))
    return weights

###############################################################################
