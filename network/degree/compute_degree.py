import numpy as np
from scipy import stats as st
import networkx as nx

save_dir = '/Users/senthilp/Desktop/Wen/degree/npy80'
cases = f'{save_dir}/cases.txt'
with open(cases) as f:
     case_list = f.read().splitlines()

n = 268
degree_count = np.zeros((n))
btw_count = np.zeros((n))
sp_count = np.zeros((n))
clc_count = np.zeros((n))

for sub in case_list:
    
    print(f'Processing {sub}...')
    conn_file = f'{save_dir}/{sub}_shen_corr_16Hz_lcmv.npy'
    connectivity = np.load(conn_file)
    connectivity = np.array(connectivity)
    result = np.zeros((connectivity.shape))
    
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
    
    degree_count = degree_count + np.sum(result, axis=1)
    np.save(f'{save_dir}/{sub}_degree.npy', result)
    
    # Betweeness
    G = nx.from_numpy_array(result, create_using=nx.Graph)
    btw = nx.betweenness_centrality(G, normalized=True)
    btw_npy = np.array(list(btw.values()))
    btw_count = btw_count + btw_npy
    np.save(f'{save_dir}/{sub}_btw.npy', btw_npy)

    # Shortest path average
    all_pairs = nx.floyd_warshall(G)
    s = [sum(t.values()) for t in all_pairs.values()]
    short_path_avg = np.array(s)/n
    sp_count = sp_count + short_path_avg
    np.save(f'{save_dir}/{sub}_sp.npy', short_path_avg)

    # Cluster
    cluster_coeff = list(nx.clustering(G).values())
    cluster_coeff = np.array(cluster_coeff)
    clc_count = clc_count + cluster_coeff
    np.save(f'{save_dir}/{sub}_clc.npy', cluster_coeff)
    
k = len(case_list)
degree_avg = degree_count / k
btw_avg = btw_count / k
sp_avg = sp_count / k
clc_avg = clc_count / k
np.save(f'{save_dir}/degree_avg.npy', degree_avg)
np.save(f'{save_dir}/btw_avg.npy', btw_avg)
np.save(f'{save_dir}/sp_avg.npy', sp_avg)
np.save(f'{save_dir}/clc_avg.npy', clc_avg)
