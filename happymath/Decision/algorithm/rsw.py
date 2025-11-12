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
    
    # 根据排名重新排序权重
    # weights 数组是优化后的结果，其顺序是按照重要性从高到低排列的
    # criteria_rank 的顺序是 C1, C2, C3, ...
    # 我们需要将 weights 的值放回对应的 C1, C2, ... 的位置
    sorted_indices = np.argsort(np.argsort(criteria_rank))
    x = x[sorted_indices]
    
    return x

###############################################################################
