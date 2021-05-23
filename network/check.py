import numpy as np
import mne
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import ttest_1samp, ttest_ind

#freqs = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
freqs = [16]
glob_deg_list = []
sig_thresh = 0.05

cases = '/home/senthilp/caesar/camcan/cc700/freesurfer_output/1.txt'
subjects_dir = '/home/senthilp/caesar/camcan/cc700/freesurfer_output'
with open(cases) as f:
     case_list = f.read().splitlines()


def Average(lst):
    return sum(lst) / len(lst)

volume_spacing = 7.8
output = []
for i, frequency in enumerate(freqs):
    sub_avg = []
    sub_btw = []
    for subject in case_list:
        DATA_DIR = Path(f'{subjects_dir}', f'{subject}', 'mne_files')
        label_corr = f'{DATA_DIR}/{subject}_corr_ortho_true_{volume_spacing}_{frequency}-label.npy'
        corr = np.load(label_corr)
        degree = corr[0, :, :].copy()

        masked = np.zeros(degree.shape)
        q = len(degree)
        for x in range(0, q-1):
            src_1_avg = np.average(degree[x])
            src_1_std = np.std(degree[x])

            for y in range(x+1, q):
                src_2_avg = np.average(degree[y])
                src_2_std = np.std(degree[y])

                signal_one = degree[x, y]
                signal_two = degree[y, x]

                away_one = src_1_avg + (1.96 * src_1_std)
                if signal_one > away_one:
                    masked[x,y] = 1
                if signal_two > src_2_avg + (1.96 * src_2_std):
                    masked[y,x] = 1
                #computed_value = [signal_one, signal_two]
                #pop_mean = [src_1_avg, src_2_avg]
                #pop_mean = (src_1_avg + src_2_avg)  / 2.0
                #t_statistic, p_value = ttest_ind(computed_value, pop_mean, alternative='greater')
                
                '''
                The observed values should be significantly different and higher than the expected value.
                The difference is significant at p < 0.01
                '''
                #if p_value < sig_thresh:
                    #print(computed_value, src_1_avg+src_2_avg, p_value)
                #    masked[x, y] = 1
                #    masked[y, x] = 1

        G = nx.from_numpy_array(masked, create_using=nx.DiGraph)
        btw = nx.betweenness_centrality(G, normalized=True)
        btw_lst = list(btw.values())
        btw_arr = np.array(btw_lst)
        likely = np.mean(btw_arr) + (1.96 * np.std(btw_arr))
        btw_sig = np.sum(btw_arr > likely)


        global_degree = np.sum(masked)

        all_conn = (masked.shape[0] * masked.shape[0]) - masked.shape[0]

        sub_avg.append(global_degree/all_conn)
        sub_btw.append(btw_sig/masked.shape[0])

    degree_global_percent = np.round(Average(sub_avg)*100, 2)
    btw_global_percent = np.round(Average(sub_btw)*100, 2)

    glob_deg_list.append(degree_global_percent)
    print(f'Frequency {frequency} {degree_global_percent} {btw_global_percent}')
    output.append(degree_global_percent)
print(output)





# group = []
# age_lst = ['18to29', '30to39', '40to49']
# glob_deg_list_18to29 = [4.11, 3.99, 3.81, 3.7, 3.52, 3.54, 3.77, 4.42, 4.61, 5.14, 4.34, 4.39, 4.16]
# glob_deg_list_30to39 = [4.04, 3.9, 3.69, 3.58, 3.45, 3.31, 3.35, 4.08, 4.56, 5.17, 4.31, 4.3, 4.13]
# glob_deg_list_40to49 = [3.84, 3.75, 3.63, 3.46, 3.23, 3.07, 3.14, 3.93, 4.54, 5.09, 4.48, 4.5, 4.34]

# group.append(glob_deg_list_18to29)
# group.append(glob_deg_list_30to39)
# group.append(glob_deg_list_40to49)

# freqs = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]

# from scipy.interpolate import make_interp_spline
# fig, ax = plt.subplots(figsize=(6, 3))

# x_pnts = np.arange(len(freqs))
# y_percent = [0, 2, 4, 6]

# for i in range(len(age_lst)):
#     y_pnts = group[i]
#     X_Y_Spline = make_interp_spline(x_pnts, y_pnts)
#     X_ = np.linspace(min(x_pnts), max(x_pnts), 500)
#     Y_ = X_Y_Spline(X_)
#     ax.plot(X_, Y_, label=age_lst[i])


# ax.set_xticks(x_pnts)
# ax.set_yticks(y_percent)
# ax.set_xticklabels(freqs, fontsize=4)
# ax.set_yticklabels(y_percent, fontsize=4)
# ax.set_xlabel('Carrier frequency (Hz)', fontsize=4)
# ax.set_ylabel('No. of connections (%)', fontsize=4)
# ax.set_title(f'Degree', fontsize=4)
# ax.legend(fontsize=8)
# plt.savefig('/home/senthilp/Desktop/degree_global.png', dpi=600)
