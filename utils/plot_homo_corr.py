import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

cases = '/home/senthil/caesar/camcan/cc700/freesurfer_output/18to30.txt'
subjects_dir = '/home/senthil/caesar/camcan/cc700/freesurfer_output'
flag = 'true'
with open(cases) as f:
     case_list = f.read().splitlines()


freq = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
y_corr = [0, 0.04, 0.08, 0.12, 0.16]
spacing=7.8
sensory_mean = {'scLeft':None, 'acLeft':None, 'vcLeft':None,
                'scRight':None, 'acRight':None, 'vcRight':None}
fig, ax = plt.subplots(figsize=(6, 3))
x_pnts = np.arange(len(freq))

mean_dict = []
for label in sensory_mean:
    myDict = {key: [] for key in freq}
    for subject in case_list:
        DATA_DIR = Path(f'{subjects_dir}', f'{subject}', 'mne_files')
        for val in freq:
            corr_data_file = f'{subjects_dir}/{subject}/mne_files/{subject}_'\
                            f'corr_ortho_{flag}_{spacing}_{val}_{label}_wholebrain.npy'
            if Path(corr_data_file).exists():
                if label == 'scLeft':
                    corr_data = float(np.load(corr_data_file)[1])*1.73
                elif label == 'acLeft':
                    corr_data = float(np.load(corr_data_file)[3])*1.73
                elif label == 'vcLeft':
                    corr_data = float(np.load(corr_data_file)[5])*1.73   
                elif label == 'scRight':
                    corr_data = float(np.load(corr_data_file)[0])*1.73
                elif label == 'acRight':
                    corr_data = float(np.load(corr_data_file)[2])*1.73
                elif label == 'vcRight':
                    corr_data = float(np.load(corr_data_file)[4])*1.73   
                myDict[val].append(corr_data)
    mean_dict.append(myDict)
    sensory_mean[label] = [np.mean(value) for key, value in myDict.items()]
    

corr_sc = [(a + b)/2 for a, b in zip(sensory_mean['scLeft'], sensory_mean['scRight'])]
corr_ac = [(a + b)/2 for a, b in zip(sensory_mean['acLeft'], sensory_mean['acRight'])]
corr_vc = [(a + b)/2 for a, b in zip(sensory_mean['vcLeft'], sensory_mean['vcRight'])]

ax.plot(x_pnts, corr_ac, '-ok', color='red', markerfacecolor='black', 
        label='Auditory', alpha=1, markersize=3, linewidth=0.5)
ax.plot(x_pnts, corr_sc, '-ok', color='gold', markerfacecolor='black', 
        label='Somat.', alpha=1, markersize=3, linewidth=0.5)
ax.plot(x_pnts, corr_vc, '-ok', color='blue', markerfacecolor='black', 
        label='Visual', alpha=1, markersize=3, linewidth=0.5)

error_bar = False
if error_bar:
    linestyle = {"linestyle":"--", "linewidth":15, "markeredgewidth":5, 
                  "elinewidth":5, "capsize":10, "alpha":0.2}
    sem_color = {'sc':"gold", 'ac':"red", 'vc':"blue"}
    for k, label in enumerate(sem_color):
        for i, val in enumerate(freq):
            ax.errorbar(x_pnts[i], mean_dict[k][val],color=sem_color[label], **linestyle)

ax.set_xticks(x_pnts)
ax.set_yticks(y_corr)
ax.set_xticklabels(freq, fontsize=4)
ax.set_yticklabels(y_corr, fontsize=4)
ax.set_xlabel('Carrier frequency (Hz)', fontsize=4)
ax.set_ylabel('Correlation', fontsize=4)
ax.set_title(f'Correlation between orthogonalized '
             f'spontaneous signals from homologous early ' 
             f'sensory areas - 72 subjects, age (18-30)', fontsize=4)
ax.legend(fontsize=8)
ax.grid(False)
plt.savefig('/home/senthil/Desktop/correlation_72.png', dpi=600)