import numpy as np
from scipy import stats as st

connectivity = np.load('shen_corr_5Hz_lcmv.npy')

connectivity = np.array(connectivity)
result = np.zeros((connectivity.shape))
total_connection = len(connectivity) - 1

for i, row in enumerate(connectivity):
    
    for j, value in enumerate(row):
            
        '''
        * 'less': the mean of the underlying distribution of the sample is
              significantly less than the given population mean (`value`)
        '''
        stat1, pvalue1 = st.ttest_1samp(row, value, alternative='less')
        stat2, pvalue2 = st.ttest_1samp(connectivity[j], value, alternative='less')
    
        if pvalue1 < 0.01 and pvalue2 < 0.01:
        
            result[i][j] = 1

threshold = np.zeros((result.shape))
upper_ind = np.triu_indices_from(result)
for x, y in zip(upper_ind[0], upper_ind[1]):   
    if result[x][y] == 1 and result[y][x] == 1:
        threshold[x][y] = 1
        threshold[y][x] = 1
    
degree_count = np.sum(threshold, axis=1)
degree_avg = degree_count / total_connection

import networkx as nx
G = nx.from_numpy_array(threshold, create_using=nx.Graph)
btw = nx.betweenness_centrality(G, normalized=True)
btw_npy = np.array(list(btw.values()))

# Shortest path average
n = 268
all_pairs = nx.floyd_warshall(G)
s = [sum(t.values()) for t in all_pairs.values()]
short_path_avg = np.array(s)/n

# Cluster
cluster_coeff = list(nx.clustering(G).values())
cluster_coeff = np.array(cluster_coeff)

# Small world
sigma = nx.sigma(G, niter=10)
