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

cases = '/home/senthilp/caesar/camcan/cc700/freesurfer_output/18to29.txt'
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
    big = np.zeros((0,190,190))
    for i, frequency in enumerate(freqs):
        label_corr = f'{DATA_DIR}/{subject}_corr_ortho_true_{volume_spacing}_{frequency}-label.npy'
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

    
    img = big.copy()
    new_img = img.reshape((img.shape[0]*img.shape[1]), img.shape[2])
    new_img = new_img.transpose()
    normalized_arr = preprocessing.normalize(new_img, norm='l2')
    norm_degree = normalized_arr.reshape(13,190,190)


    sub_avg = []
    sub_btw = []

    for i, degree in enumerate(norm_degree):

        masked_degree = f'{DATA_DIR}/{subject}_degreeMasked_{volume_spacing}_{freqs[i]}-label.npy'

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

                signal_one = big[i][x,y]
                signal_two = big[i][y,x]


                signi_mean = (src_1_avg + src_2_avg) / 2
                signi_std = (src_1_std + src_2_std) / 2

                away = signi_mean + (1.96 * signi_std)

                if signal_one > away:
                    masked[x,y] = 1
                if signal_two > away:
                    masked[y,x] = 1


        np.save(masked_degree, masked)
        print(masked_degree)
        G = nx.from_numpy_array(masked, create_using=nx.DiGraph)
        btw = nx.betweenness_centrality(G, normalized=True)
        btw_lst = list(btw.values())
        btw_arr = np.array(btw_lst)
        likely = np.mean(btw_arr) + (1.96 * np.std(btw_arr))
        btw_sig = np.sum(btw_arr > likely)


        global_degree = np.sum(masked)
        all_conn = (degree.shape[0] * degree.shape[0]) - degree.shape[0]

        cal = (global_degree/all_conn)
        degree_global_percent = np.round((cal*100),2)

        sub_avg.append(degree_global_percent)

        cal2 = btw_sig/degree.shape[0]
        btw_global_percent = np.round((cal2*100),2)
        sub_btw.append(btw_global_percent)

        print(f'Frequency: {freqs[i]} | degree: {degree_global_percent}%')
    output.append(sub_avg)
    output_btw.append(sub_btw)

avg = [float(sum(col))/len(col) for col in zip(*output)]
avg_btw = [float(sum(col))/len(col) for col in zip(*output_btw)]
print(avg)
print(avg_btw)
