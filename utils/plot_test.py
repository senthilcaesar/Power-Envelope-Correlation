import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 4})

cases = '/home/senthilp/caesar/camcan/cc700/freesurfer_output/18to30.txt'
subjects_dir = '/home/senthilp/caesar/camcan/cc700/freesurfer_output'
flag = 'true'
with open(cases) as f:
     case_list = f.read().splitlines()


freq = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
y_corr = [0, 0.04, 0.08, 0.12, 0.16]
spacing=30
sensory_mean = {'sc':None, 'ac':None, 'vc':None}
fig, ax = plt.subplots(figsize=(7, 3))


mean_dict = []
for label in sensory_mean:
    corr_data = np.zeros((105,))
    for subject in case_list:
        DATA_DIR = Path(f'{subjects_dir}', f'{subject}', 'mne_files')
        corr_data_file = f'{subjects_dir}/{subject}/mne_files/{subject}_'\
                            f'coh_{spacing}_{label}_epoch.npy'
        corr_data = corr_data + np.load(corr_data_file)
    sensory_mean[label] = corr_data / len(case_list)
    

freq = np.load('/home/senthilp/Downloads/freqs.npy')

ax.plot(sensory_mean['ac'], '-ok', color='red', markerfacecolor='black', 
        label='Auditory', alpha=1, markersize=3, linewidth=0.5)
ax.plot(sensory_mean['sc'], '-ok', color='gold', markerfacecolor='black', 
        label='Somat.', alpha=1, markersize=3, linewidth=0.5)
ax.plot(sensory_mean['vc'], '-ok', color='blue', markerfacecolor='black', 
        label='Visual', alpha=1, markersize=3, linewidth=0.5)


ax.set_xticks([0,10,20,30,40,50,60,70,80,90,100])
ax.set_xticklabels(['2','14','26','38','50','62', 
                    '74','86', '98','110', '122'], fontsize=4)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Coherence')
ax.set_title(f'Coherence between analytic '
             f'signals from homologous early ' 
             f'sensory areas - 72 subjects, age (18-30)')
ax.legend()
ax.grid(False)
plt.savefig('/home/senthilp/Desktop/coherence_72.png', dpi=600)
