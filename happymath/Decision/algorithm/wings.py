###############################################################################

# Required Libraries

import numpy as np

###############################################################################

# Function: WINGS (Weighted Influence Non-linear Gauge System)
def wings(dataset): 
    D = dataset
    C = D/np.sum(D)
    I = np.eye(dataset.shape[0])
    T = np.dot(C, np.linalg.inv( I - C ))
    c_i = np.sum(T, axis = 0)
    r_i = np.sum(T, axis = 1)
    r_plus_c  = r_i + c_i 
    r_minus_c = r_i - c_i 
    weights   = r_plus_c/np.sum(r_plus_c)
    xmin = np.amin(r_plus_c)
    if (xmin > 0):
        xmin = 0
    xmax = np.amax(r_plus_c)
    if (xmax < 0):
        xmax = 0
    ymin = np.amin(r_minus_c)
    if (ymin > 0):
        ymin = 0
    ymax = np.amax(r_minus_c)
    if (ymax < 0):
        ymax = 0
    return r_plus_c, r_minus_c, weights

###############################################################################