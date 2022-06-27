import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats as st


def without_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}


def participation_coefficient(G, module_partition):

    pc_dict = {}
    for m in module_partition.keys():
        M = set(module_partition[m])
        for v in M:
            degree = float(G.out_degree(v))
            wm_degree = float(sum([1 for u in M if (u, v) in G.edges()]))
            pc_dict[v] = 1 - ((float(wm_degree) / float(degree))**2)
    return pc_dict


def adjacency_matrix(corr):

    result = np.zeros((corr.shape))
    for i, row in enumerate(corr):  
        for j, value in enumerate(row):
            stat1, pvalue1 = st.ttest_1samp(row, value, alternative='less')
            stat2, pvalue2 = st.ttest_1samp(corr[j], value, alternative='less')
            if pvalue1 < 0.01 and pvalue2 < 0.01:
                result[i][j] = 1
    return result


# System parition
ntw_map = '/Users/senthilp/Desktop/wen/degree/power264NodesNetwork.xlsx'
df = pd.read_excel(ntw_map)
df['label'] = df['Name'].str.replace('\d+', '', regex=True)
df['label'] = df['label'].str.rstrip('_')
df['ID'] = df['ID'] - 1
g = df.groupby('Network')['ID'].apply(lambda x: list(np.unique(x)))
module_partition = g.to_dict()
module_partition = without_keys(module_partition,[6,10,13,14]) # Ignore systems

# Load correlation
fname = '/Users/senthilp/Downloads/power_corr_4Hz_lcmv.npy'
corr = np.load(fname)
AM = adjacency_matrix(corr)
G = nx.from_numpy_array(AM, create_using=nx.DiGraph)
pc_dict = participation_coefficient(G, module_partition)
mean_ntw_PC = sum(pc_dict.values()) / len(pc_dict)
print(mean_ntw_PC)
