import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


freqs = {
    4: [2, 6],
    6: [4, 8],
    8: [6, 10],
    12: [10, 14],
    16: [14, 18],
    24: [22, 26],
    32: [30, 34],
    48: [46, 50],
    64: [62, 66],
    96: [94, 98],
    128: [126, 130]
}

cases = '/home/senthil/caesar/camcan/cc700/freesurfer_output/age18to30.txt'
subjects_dir = '/home/senthil/caesar/camcan/cc700/freesurfer_output'
with open(cases) as f:
     case_list = f.read().splitlines()


sensory_mean = {'ac':None , 'sc':None, 'vc':None}

space = 30
covar_freq_list = []
for freq in freqs:
    mean_coh = np.zeros([26, 1])
    for label in sensory_mean:
        for subject in case_list:
            DATA_DIR = Path(f'{subjects_dir}', f'{subject}', 'mne_files')
            corr_data_file = f'{DATA_DIR}/{subject}_coh_{space}_{freq}_{label}.npy'
            mean_coh = mean_coh + np.load(corr_data_file)
    mean_coh = mean_coh / (len(case_list)*3)
    covar_freq_list.append(mean_coh)
    
coherence_correlation = np.hstack(covar_freq_list)
coherence_correlation = np.swapaxes(coherence_correlation, 0, 1)
coherence_correlation = np.flip(coherence_correlation, 0)

fig, ax = plt.subplots(figsize=(7, 3))
plt.imshow(coherence_correlation, aspect='auto', 
            cmap='jet', interpolation='gaussian') #vmin=-0.5, vmax=0.5)
ax.set_xticks([0,5,10,15,20,25])
ax.set_xticklabels(['0.032','0.1','0.32','1.0','3.2','10'], fontsize=4)
ax.set_yticks([0,1,2,3,4,5,6,7,8,9,10])
ax.set_yticklabels(['128', '96', '64','48','32','24','16','12'
                    ,'8', '6', '4'], fontsize=4)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(False)
ax.set_xlabel('Co-variation frequency (Hz)', fontsize=6)
ax.set_ylabel('Carrier frequency (Hz)', fontsize=6)
plt.colorbar()
plt.savefig('/home/senthil/Desktop/covar_80.png', dpi=600)
