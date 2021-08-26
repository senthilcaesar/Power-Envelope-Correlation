import numpy as np
import mne
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import ttest_1samp, ttest_ind
from sklearn.preprocessing import RobustScaler
from scipy import stats
from sklearn import preprocessing

freqs = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
glob_deg_list = []

cases = '/home/senthilp/caesar/camcan/cc700/freesurfer_output/remain.txt'
subjects_dir = '/home/senthilp/caesar/camcan/cc700/freesurfer_output'
with open(cases) as f:
     case_list = f.read().splitlines()


def Average(lst):
    return sum(lst) / len(lst)

volume_spacing = 7.8
output = []
output_btw = []

for subject in case_list:
    DATA_DIR = Path(f'{subjects_dir}', f'{subject}', 'mne_files')
    for i, frequency in enumerate(freqs):
        label_corr = f'scp -pr senthilp@128.197.62.176:{DATA_DIR}/{subject}_corr_ortho_true_{volume_spacing}_{frequency}_label.npy .'
        print(label_corr)
