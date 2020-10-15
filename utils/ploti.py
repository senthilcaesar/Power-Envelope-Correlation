import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

cases = '/home/senthil/caesar/camcan/cc700/freesurfer_output/10.txt'
subjects_dir = '/home/senthil/caesar/camcan/cc700/freesurfer_output'
flag = 'True'
with open(cases) as f:
     case_list = f.read().splitlines()


freq = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20] #, 24, 28, 32, 42, 52, 64, 96, 128]
myDict = {key: [] for key in freq}
sensory = {'sc':None, 'ac':None, 'vc':None}

for label in sensory:
    for subject in case_list:
        DATA_DIR = Path(f'{subjects_dir}', f'{subject}', 'mne_files')
        for val in freq:
            corr_data_file = f'{DATA_DIR}/{subject}_corr_ortho_true_5_{val}_{label}.npy'
            if Path(corr_data_file).exists():
                corr_data = float(np.load(corr_data_file))
                myDict[val].append(corr_data)
    sensory[label] = [np.mean(value) for key, value in myDict.items()]
    

x_pnts = np.arange(len(freq))
x_values = [2, 4, 8, 16, 32, 64, 128]
y_corr = [0, 0.04, 0.08, 0.12, 0.16]

fig, ax = plt.subplots(figsize=(5, 3))

ax.plot(x_pnts, sensory['ac'], '-ok', color='red', markerfacecolor='black', label='Auditory')
ax.plot(x_pnts, sensory['sc'], '-ok', color='gold', markerfacecolor='black', label='Somat')
ax.plot(x_pnts, sensory['vc'], '-ok', color='blue', markerfacecolor='black', label='Visual')

ax.set_xticks(x_pnts)
ax.set_yticks(y_corr)
ax.set_xticklabels(freq)
ax.set_yticklabels(y_corr)

# xticks = ax.xaxis.get_major_ticks()
# xticks[2].label1.set_visible(False)
# xticks[4].label1.set_visible(False)
# xticks[5].label1.set_visible(False)
# xticks[6].label1.set_visible(False)
      
ax.set_xlabel('Carrier frequency (Hz)')
ax.set_ylabel('Correlation')
leg = ax.legend();