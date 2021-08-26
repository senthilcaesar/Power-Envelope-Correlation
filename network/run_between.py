import numpy as np
import mne
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import ttest_1samp, ttest_ind
from sklearn.preprocessing import RobustScaler
from scipy import stats
from sklearn import preprocessing

freqs = [6]#, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
glob_deg_list = []

cases = '/home/senthilp/caesar/camcan/cc700/freesurfer_output/80to88.txt'
subjects_dir = '/home/senthilp/caesar/camcan/cc700/freesurfer_output'
with open(cases) as f:
     case_list = f.read().splitlines()

def Average(lst):
    return sum(lst) / len(lst)

volume_spacing = 7.8
output = []
sub_btw = []


for i, frequency in enumerate(freqs):
    perusu = []
    DATA_DIR = Path(f'{subjects_dir}', 'data', 'degreeMasked')
    for subject in case_list:
        gilla = []
        degree_bin = f'{DATA_DIR}/{subject}_degreeMasked_7.8_{frequency}.npy'
        print(degree_bin)
        masked_arr = np.load(degree_bin)
        if np.sum(masked_arr) > 5:
        
            for i in range(0, 1000):
                masked = np.random.permutation(masked_arr).copy()
                G = nx.from_numpy_array(masked, create_using=nx.DiGraph)
                btw = nx.betweenness_centrality(G, normalized=True)
                btw_lst = list(btw.values())
                btw_arr = np.array(btw_lst)
                gilla.append(btw_arr.max())
            perusu.append(gilla)
    
    avg = [np.round(float(sum(col))/len(col), 2) for col in zip(*perusu)]
    avg = np.array(avg)
    away = np.mean(avg) + (1.96 * np.std(avg))
    
    avg[avg > away] = 1
    avg[avg <= away] = 0
    print((np.sum(avg)/len(avg))*100) 

