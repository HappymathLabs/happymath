###############################################################################

# Required Libraries
import numpy as np

###############################################################################

# Function: ROC (Rank Ordered Centroid)
def roc(criteria_rank):
    
    ################################################
    
    x = np.zeros(len(criteria_rank))
    for i in range(0, x.shape[0]):
        for j in range(i, x.shape[0]):
            x[i] = x[i] + 1/(j+1)
    x   = x/len(criteria_rank)
    x   = x/np.sum(x)
    
    sorted_indices = np.argsort(np.argsort(criteria_rank))
    x = x[sorted_indices]
    
    return x

###############################################################################
