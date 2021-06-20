
import numpy as np
import mne
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import ttest_1samp, ttest_ind


group = []
age_lst = ['18to29']
glob_deg_list_18to29 = [14.125820895522386, 12.925970149253734, 12.318507462686565, 15.675671641791041, 28.764477611940304, 30.33955223880597, 15.635522388059703, 4.640746268656714, 1.4089552238805967, 25.00820895522388, 2.4459701492537307, 3.056119402985076, 3.643134328358209]


group.append(glob_deg_list_18to29)


freqs = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]

from scipy.interpolate import make_interp_spline
fig, ax = plt.subplots(figsize=(6, 3))

x_pnts = np.arange(len(freqs))
y_percent = [0, 5, 10, 15, 20, 25, 30, 35]

for i in range(len(age_lst)):
    y_pnts = group[i]
    X_Y_Spline = make_interp_spline(x_pnts, y_pnts)
    X_ = np.linspace(min(x_pnts), max(x_pnts), 500)
    Y_ = X_Y_Spline(X_)
    ax.plot(X_, Y_, label=age_lst[i])


ax.set_xticks(x_pnts)
ax.set_yticks(y_percent)
ax.set_xticklabels(freqs, fontsize=4)
ax.set_yticklabels(y_percent, fontsize=4)
ax.set_xlabel('Carrier frequency (Hz)', fontsize=4)
ax.set_ylabel('No. of connections (%)', fontsize=4)
ax.set_title(f'Degree', fontsize=4)
ax.legend(fontsize=8)
plt.savefig('/home/senthilp/Desktop/degree_global_young.png', dpi=600)