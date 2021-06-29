import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

cases = '/home/senthilp/caesar/camcan/cc700/freesurfer_output/80to88.txt'
subjects_dir = '/home/senthilp/caesar/camcan/cc700/freesurfer_output'
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
                            f'corr_ortho_{flag}_{spacing}_{val}_{label}.npy'
            if Path(corr_data_file).exists():
                if label == 'scLeft':
                    corr_data = float(np.load(corr_data_file)[1]) * 1.73
                elif label == 'acLeft':
                    corr_data = float(np.load(corr_data_file)[3]) * 1.73
                elif label == 'vcLeft':
                    corr_data = float(np.load(corr_data_file)[5]) * 1.73   
                elif label == 'scRight':
                    corr_data = float(np.load(corr_data_file)[0]) * 1.73
                elif label == 'acRight':
                    corr_data = float(np.load(corr_data_file)[2]) * 1.73
                elif label == 'vcRight':
                    corr_data = float(np.load(corr_data_file)[4]) * 1.73   
                myDict[val].append(corr_data)
    mean_dict.append(myDict)
    sensory_mean[label] = [np.mean(value) for key, value in myDict.items()]
    

mean_corr_sc = [(a + b)/2 for a, b in zip(sensory_mean['scLeft'], sensory_mean['scRight'])]
mean_corr_ac = [(a + b)/2 for a, b in zip(sensory_mean['acLeft'], sensory_mean['acRight'])]
mean_corr_vc = [(a + b)/2 for a, b in zip(sensory_mean['vcLeft'], sensory_mean['vcRight'])]

ax.plot(x_pnts, mean_corr_ac, '-ok', color='red', markerfacecolor='black', 
        label='Auditory', alpha=1, markersize=3, linewidth=0.5)
ax.plot(x_pnts, mean_corr_sc, '-ok', color='gold', markerfacecolor='black', 
        label='Somat.', alpha=1, markersize=3, linewidth=0.5)
ax.plot(x_pnts, mean_corr_vc, '-ok', color='blue', markerfacecolor='black', 
        label='Visual', alpha=1, markersize=3, linewidth=0.5)

error_bar = False
if error_bar:
    linestyle = {"linestyle":"--", "linewidth":15, "markeredgewidth":5, 
                  "elinewidth":5, "capsize":10, "alpha":0.2}
    sem_color = {'sc':"gold", 'ac':"red", 'vc':"blue"}
    for k, label in enumerate(sem_color):
        for i, val in enumerate(freq):
            ax.errorbar(x_pnts[i], mean_dict[k][val],color=sem_color[label], **linestyle)

def compute_sem(mean_dict):
    from scipy import stats
    work = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
    for i, roi in enumerate(mean_dict):
        for each_freq in roi:
            work[i].append(stats.sem(roi[each_freq]))
        
    sem_sc = [(a + b)/2 for a, b in zip(work[0], work[3])]
    sem_ac = [(a + b)/2 for a, b in zip(work[1], work[4])]
    sem_vc = [(a + b)/2 for a, b in zip(work[2], work[5])]

    return sem_sc, sem_ac, sem_vc

sem_sc, sem_ac, sem_vc = compute_sem(mean_dict)

plus_sc = [(a + b) for a, b in zip(mean_corr_sc, sem_sc)]
plus_ac = [(a + b) for a, b in zip(mean_corr_ac, sem_ac)]
plus_vc = [(a + b) for a, b in zip(mean_corr_vc, sem_vc)]

minus_sc = [(a - b) for a, b in zip(mean_corr_sc, sem_sc)]
minus_ac = [(a - b) for a, b in zip(mean_corr_ac, sem_ac)]
minus_vc = [(a - b) for a, b in zip(mean_corr_vc, sem_vc)]

ax.fill_between(x_pnts, plus_ac, minus_ac, color='red', alpha=0.1)
ax.fill_between(x_pnts, plus_sc, minus_sc, color='gold', alpha=0.1)
ax.fill_between(x_pnts, plus_vc, minus_vc, color='blue', alpha=0.1)

ax.set_xticks(x_pnts)
ax.set_yticks(y_corr)
ax.set_xticklabels(freq, fontsize=4)
ax.set_yticklabels(y_corr, fontsize=4)
ax.set_xlabel('Carrier frequency (Hz)', fontsize=4)
ax.set_ylabel('Correlation', fontsize=4)
ax.set_title(f'Correlation between orthogonalized '
              f'spontaneous signals from homologous early ' 
              f'sensory areas - age (80-88)', fontsize=4)
ax.legend(fontsize=8)
ax.grid(False)
plt.savefig('/home/senthilp/Desktop/80to88.png', dpi=600)
