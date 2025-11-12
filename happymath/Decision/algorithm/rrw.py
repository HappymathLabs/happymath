###############################################################################

# Required Libraries
import numpy as np

###############################################################################

# Function: RRW (Rank Reciprocal Weighting)
def rrw(criteria_rank):
    
    ################################################
    
    S = 0
    x = np.zeros(len(criteria_rank))
    for i in range(0, x.shape[0]):
        S = S + 1/(i+1) 
    for i in range(0, x.shape[0]):
        x[i] = 1 / ( (i+1) * S)
    x   = x/np.sum(x)
    
    sorted_indices = np.argsort(np.argsort(criteria_rank))
    x = x[sorted_indices]
    
    return x

###############################################################################
