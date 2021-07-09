#%%
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv

cases = '/home/senthilp/caesar/camcan/cc700/freesurfer_output/80to88.txt'
subjects_dir = '/home/senthilp/caesar/camcan/cc700/freesurfer_output'
with open(cases) as f:
     case_list = f.read().splitlines()

log_range = np.arange(2,7.25,0.25)
carrier_freqs = [math.pow(2,val) for val in log_range]

sensory_mean = {'sc':None, 'ac':None, 'vc':None}

space = 30
covar_freq_list = []
for freq in carrier_freqs:
    mean_coh = np.zeros([26, 1])
    for label in sensory_mean:
        for subject in case_list:
            DATA_DIR = Path(f'{subjects_dir}', f'{subject}', 'mne_files')
            corr_data_file = f'{DATA_DIR}/{subject}_coh_{space}_{freq}_{label}.npy'
            mean_coh = mean_coh + np.load(corr_data_file)
    mean_coh = mean_coh / (len(case_list)*len(sensory_mean))
    covar_freq_list.append(mean_coh)
    
coherence_correlation = np.hstack(covar_freq_list)
coherence_correlation = np.swapaxes(coherence_correlation, 0, 1)
coherence_correlation = np.flip(coherence_correlation, 0)

fig, ax = plt.subplots(figsize=(6, 3))
plt.imshow(coherence_correlation, aspect='auto', 
            cmap='jet', interpolation='gaussian', vmin=0.5, vmax=1)
ax.set_xticks([0,5,10,15,20,25])
ax.set_xticklabels(['0.032','0.1','0.32','1.0','3.2','10'], fontsize=4)
ax.set_yticks([0,4,8,12,16,20])
ax.set_yticklabels(['128', '64','32','16','8','4'], fontsize=4)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(False)
ax.set_xlabel('Co-variation frequency (Hz)', fontsize=6)
ax.set_ylabel('Carrier frequency (Hz)', fontsize=6)
ax.set_title(f'Correlation between homologous sensory '
             f'areas as a function of the carrier frequency and  ' 
             f'the co-variation frequency - age (80-88)', fontsize=4)
plt.colorbar()
plt.savefig('/home/senthilp/Desktop/covar_80to88.png', dpi=600)
# %%
