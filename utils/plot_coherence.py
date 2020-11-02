import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

cases = '/home/senthil/caesar/camcan/cc700/freesurfer_output/50.txt'
subjects_dir = '/home/senthil/caesar/camcan/cc700/freesurfer_output'
with open(cases) as f:
     case_list = f.read().splitlines()


sensory_mean = {'ac':None , 'sc':None, 'vc':None}
fig, ax = plt.subplots(figsize=(6, 3))

spacing = 30
freq = np.load('/home/senthil/Downloads/freq.npy')
mean_coh = np.zeros([26, 275])

for label in sensory_mean:
    for subject in case_list:
        DATA_DIR = Path(f'{subjects_dir}', f'{subject}', 'mne_files')
        corr_data_file = f'{DATA_DIR}/{subject}_coh_{spacing}_{label}.npy'
        mean_coh = mean_coh + np.load(corr_data_file)[:,:275]

mean_coh = mean_coh / (len(case_list)*3)
mean_coh = mean_coh.T
mean_coh = np.flip(mean_coh, 0)

x_label = [0.032, 0.05, 0.1, 0.32, 1.0, 3.2, 10]
y_label = [256, 128, 64, 32, 16, 8, 4]

plt.imshow(mean_coh, aspect='auto', 
            cmap='jet', interpolation='gaussian',
            vmin=-0.5, vmax=0.5)
ax.set_xticklabels(x_label,fontsize=4)
ax.set_yticklabels(y_label,fontsize=4)
ax.set_xlabel('Co-variation frequency (Hz)', fontsize=6)
ax.set_ylabel('Carrier frequency (Hz)', fontsize=6)
plt.colorbar()
plt.savefig('/home/senthil/Desktop/covar_one.png', dpi=600)
