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

cases = '/home/senthilp/caesar/camcan/cc700/freesurfer_output/80to88.txt'
subjects_dir = '/home/senthilp/caesar/camcan/cc700/freesurfer_output'
with open(cases) as f:
     case_list = f.read().splitlines()


def Average(lst):
    return sum(lst) / len(lst)

volume_spacing = 7.8
output = []
output_btw = []

for subject in case_list:
    print(f'{subject}---------------------------------------------------------')
    DATA_DIR = Path(f'{subjects_dir}', 'data')
    big = np.zeros((0,190,190))
    for i, frequency in enumerate(freqs):
        label_corr = f'{DATA_DIR}/{subject}_corr_ortho_true_{volume_spacing}_{frequency}_label.npy'
        #print(label_corr)
        corr = np.load(label_corr)

        if corr.shape[1] == 190:
            degree = corr[0, :, :].copy()
            degree = np.expand_dims(degree, axis=0)
            big = np.append(big, degree, axis=0)
            big[big > 0.1] = 0.1
        else:
            degree = np.zeros((190,190))
            degree = np.expand_dims(degree, axis=0)
            big = np.append(big, degree, axis=0)

    sub_avg = []

    for i, degree in enumerate(big):
        masked_degree = f'{DATA_DIR}/degreeMasked/{subject}_degreeMasked_{volume_spacing}_{freqs[i]}.npy'

        q = len(degree)
        masked = np.zeros((q,q))
        for x in range(0, q-1):
            label1 = degree[x]
            src_1_avg = np.mean(label1)
            src_1_std = np.std(label1)

            for y in range(x+1, q):
                label2 = degree[y]
                src_2_avg = np.mean(label2)
                src_2_std = np.std(label2)

                signal_one = degree[x,y]
                signal_two = degree[y,x]


                signi_mean = (src_1_avg + src_2_avg) / 2
                signi_std = (src_1_std + src_2_std) / 2

                away = 0.04 #signi_mean + (1.96 * signi_std)

                if signal_one > away:
                    masked[x,y] = 1
                if signal_two > away:
                    masked[y,x] = 1


        np.save(masked_degree, masked)
        global_degree = np.sum(masked)
        all_conn = (degree.shape[0] * degree.shape[0]) - degree.shape[0]

        cal = (global_degree/all_conn)
        degree_global_percent = np.round((cal*100),2)

        sub_avg.append(degree_global_percent)
        print(f'Frequency: {freqs[i]} | degree: {degree_global_percent}%')

    output.append(sub_avg)

avg = [np.round(float(sum(col))/len(col), 2) for col in zip(*output)]
print(avg)

