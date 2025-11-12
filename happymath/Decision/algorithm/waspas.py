###############################################################################

# Required Libraries
import matplotlib.pyplot as plt
import numpy as np

###############################################################################

# Function: Rank 
def ranking_m(flow_1, flow_2, flow_3):    
    rank_xy = np.zeros((flow_1.shape[0] + 1, 6)) 
    for i in range(0, rank_xy.shape[0]):
        rank_xy[i, 0] = -1
        rank_xy[i, 1] = flow_1.shape[0]-i+1     
        rank_xy[i, 2] = 0
        rank_xy[i, 3] = flow_2.shape[0]-i+1  
        rank_xy[i, 4] = 1
        rank_xy[i, 5] = flow_3.shape[0]-i+1    
    plt.text(rank_xy[0, 0],  rank_xy[0, 1], 'WSM', size = 12, ha = 'center', va = 'center', color = 'white', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0, 0, 0),))
    plt.text(rank_xy[0, 2],  rank_xy[0, 3], 'WPM', size = 12, ha = 'center', va = 'center', color = 'white', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0, 0, 0),))
    plt.text(rank_xy[0, 4],  rank_xy[0, 5], 'WASPAS', size = 12, ha = 'center', va = 'center', color = 'white', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0, 0, 0),))
    for i in range(1, rank_xy.shape[0]):
        plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow_1[i-1,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = "round", ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),))
        plt.text(rank_xy[i, 2],  rank_xy[i, 3], 'a' + str(int(flow_2[i-1,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),))
        plt.text(rank_xy[i, 4],  rank_xy[i, 5], 'a' + str(int(flow_3[i-1,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),)) 
    for i in range(1, rank_xy.shape[0]-1):
        plt.arrow(rank_xy[i, 0], rank_xy[i, 1], rank_xy[i+1, 0] - rank_xy[i, 0], rank_xy[i+1, 1] - rank_xy[i, 1], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
        plt.arrow(rank_xy[i, 2], rank_xy[i, 3], rank_xy[i+1, 2] - rank_xy[i, 2], rank_xy[i+1, 3] - rank_xy[i, 3], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
        plt.arrow(rank_xy[i, 4], rank_xy[i, 5], rank_xy[i+1, 4] - rank_xy[i, 4], rank_xy[i+1, 5] - rank_xy[i, 5], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
    axes = plt.gca()
    axes.set_xlim([-2, +2])
    ymin = np.amin(rank_xy[:,1])
    ymax = np.amax(rank_xy[:,1])
    if (ymin < ymax):
        axes.set_ylim([ymin, ymax])
    else:
        axes.set_ylim([ymin-1, ymax+1])
    plt.axis('off')
    plt.show() 
    return

# Function: WASPAS
def waspas(dataset, criterion_type, weights, lambda_value, graph = True, verbose = True):
    x = np.zeros((dataset.shape[0], dataset.shape[1]), dtype = float)
    for j in range(0, dataset.shape[1]):
        if (criterion_type[j] == 'max'):
            x[:,j] = 1 + ( dataset[:,j] - np.min(dataset[:,j]) ) / ( np.max(dataset[:,j]) - np.min(dataset[:,j]) )
        else:
            x[:,j] = 1 + ( np.max(dataset[:,j]) - dataset[:,j] ) / ( np.max(dataset[:,j]) - np.min(dataset[:,j]) )
    wsm_scores    = np.sum(x*weights, axis = 1)
    wpm_scores    = np.prod(x**weights, axis = 1)
    waspas_scores = (lambda_value)*wsm_scores + (1 - lambda_value)*wpm_scores
    
    num_alternatives = wsm_scores.shape[0]
    indices = np.arange(1, num_alternatives + 1).reshape(-1, 1)
    
    wsm = np.hstack((indices, wsm_scores.reshape(-1, 1)))
    
    wpm = np.hstack((indices, wpm_scores.reshape(-1, 1)))
    
    waspas = np.hstack((indices, waspas_scores.reshape(-1, 1)))
    
    if (verbose == True):
        print('WSM Scores:')
        for i in range(0, wsm.shape[0]):
            print('a' + str(int(wsm[i,0])) + ': ' + str(round(wsm[i,1], 3)))
        print('WPM Scores:')
        for i in range(0, wpm.shape[0]):
            print('a' + str(int(wpm[i,0])) + ': ' + str(round(wpm[i,1], 3)))
        print('WASPAS Final Scores:')
        for i in range(0, waspas.shape[0]):
            print('a' + str(int(waspas[i,0])) + ': ' + str(round(waspas[i,1], 3)))
    
    if (graph == True):
        ranking_m(wsm[np.argsort(wsm[:, 1])[::-1]], wpm[np.argsort(wpm[:, 1])[::-1]], waspas[np.argsort(waspas[:, 1])[::-1]])
    
    return wsm, wpm, waspas

###############################################################################

