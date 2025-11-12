###############################################################################

# Required Libraries
import numpy as np

###############################################################################

# Function: RSW (Rank Summed Weight)
def rsw(criteria_rank):
    
    ################################################
    
    N = len(criteria_rank)
    x = np.zeros(N)
    for i in range(0, x.shape[0]):
        x[i] = ( 2 * (N - (i+1) + 1) ) / ( N * (N  + 1) )
    x   = x/np.sum(x)
    
    sorted_indices = np.argsort(np.argsort(criteria_rank))
    x = x[sorted_indices]
    
    return x

###############################################################################
