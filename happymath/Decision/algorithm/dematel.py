###############################################################################

# Required Libraries

import numpy as np

###############################################################################

# Function: DEMATEL
def dematel(dataset, size_x = 10, size_y = 10):  
    row_sum = np.sum(dataset, axis = 1)
    max_sum = np.max(row_sum)
    X = dataset/max_sum
    Y = np.linalg.inv(np.identity(dataset.shape[0]) - X) 
    T = np.matmul (X, Y)
    D = np.sum(T, axis = 1)
    R = np.sum(T, axis = 0)
    D_plus_R   = D + R # Most Importante Criteria
    D_minus_R  = D - R # +Influencer Criteria, - Influenced Criteria
    weights    = D_plus_R/np.sum(D_plus_R)

    xmin = np.amin(D_plus_R)
    if (xmin > 0):
        xmin = 0
    xmax = np.amax(D_plus_R)
    if (xmax < 0):
        xmax = 0

    ymin = np.amin(D_minus_R)
    if (ymin > 0):
        ymin = 0
    ymax = np.amax(D_minus_R)
    if (ymax < 0):
        ymax = 0

    return D_plus_R, D_minus_R, weights

###############################################################################