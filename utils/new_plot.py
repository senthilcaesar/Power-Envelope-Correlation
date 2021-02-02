import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

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


freq = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
y_corr = [0, 0.04, 0.08, 0.12, 0.16]
spacing=30
flag = 'true'
sensory_mean = {'scLeft':None, 'acLeft':None, 'vcLeft':None,
                'scRight':None, 'acRight':None, 'vcRight':None}

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 5))
x_pnts = np.arange(len(freq))

color_plot = ['red', 'green', 'blue', 'gold', 'fuchsia', 'brown', 'cyan']
age = ['18to29',
       '30to39',
       '40to49',
       '50to59',
       '60to69',
       '70to79',
       '80to88']
for i, age_file in enumerate(age):
    cases = f'/home/senthilp/caesar/camcan/cc700/{age_file}.txt'
    subjects_dir = '/home/senthilp/caesar/camcan/cc700/freesurfer_output'
    
    with open(cases) as f:
         case_list = f.read().splitlines() 
         
    mean_dict = []
    for label in sensory_mean:
        myDict = {key: [] for key in freq}
        for subject in case_list:
            DATA_DIR = Path(f'{subjects_dir}', f'{subject}', 'mne_files')
            for val in freq:
                corr_data_file = f'{subjects_dir}/{subject}/mne_files/{subject}_'\
                                f'corr_ortho_{flag}_{spacing}_{val}_{label}_check.npy'
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
    
    
    sem_sc, sem_ac, sem_vc = compute_sem(mean_dict)
    plus_sc = [(a + b) for a, b in zip(mean_corr_sc, sem_sc)]
    plus_ac = [(a + b) for a, b in zip(mean_corr_ac, sem_ac)]
    plus_vc = [(a + b) for a, b in zip(mean_corr_vc, sem_vc)]
    minus_sc = [(a - b) for a, b in zip(mean_corr_sc, sem_sc)]
    minus_ac = [(a - b) for a, b in zip(mean_corr_ac, sem_ac)]
    minus_vc = [(a - b) for a, b in zip(mean_corr_vc, sem_vc)]
    
    ax1.plot(x_pnts, mean_corr_ac, '-ok', color=color_plot[i], markerfacecolor='black', 
        label=f'Auditory {age_file}', alpha=1, markersize=3, linewidth=0.5)
    ax1.fill_between(x_pnts, plus_ac, minus_ac, color=color_plot[i], alpha=0.05)
    
    ax2.plot(x_pnts, mean_corr_sc, '-ok', color=color_plot[i], markerfacecolor='black', 
        label=f'Somat {age_file}', alpha=1, markersize=3, linewidth=0.5)
    ax2.fill_between(x_pnts, plus_sc, minus_sc, color=color_plot[i], alpha=0.05)
    
    ax3.plot(x_pnts, mean_corr_vc, '-ok', color=color_plot[i], markerfacecolor='black', 
        label=f'Visual {age_file}', alpha=1, markersize=3, linewidth=0.5)
    ax3.fill_between(x_pnts, plus_vc, minus_vc, color=color_plot[i], alpha=0.05)


ax1.set_xticks(x_pnts)
ax1.set_yticks(y_corr)
ax1.set_xticklabels(freq, fontsize=8)
ax1.set_yticklabels(y_corr, fontsize=8)
ax1.set_xlabel('Carrier frequency (Hz)', fontsize=8)
ax1.set_ylabel('Correlation', fontsize=8)
ax1.set_title(f'Correlation between orthogonalized '
              f'spontaneous signals from Auditory ' 
              f'sensory areas', fontsize=8)
ax1.legend(fontsize=8)
ax1.grid(False)

ax2.set_xticks(x_pnts)
ax2.set_yticks(y_corr)
ax2.set_xticklabels(freq, fontsize=8)
ax2.set_yticklabels(y_corr, fontsize=8)
ax2.set_xlabel('Carrier frequency (Hz)', fontsize=8)
ax2.set_ylabel('Correlation', fontsize=8)
ax2.set_title(f'Correlation between orthogonalized '
              f'spontaneous signals from Somat ' 
              f'sensory areas', fontsize=8)
ax2.legend(fontsize=8)
ax2.grid(False)

ax3.set_xticks(x_pnts)
ax3.set_yticks(y_corr)
ax3.set_xticklabels(freq, fontsize=8)
ax3.set_yticklabels(y_corr, fontsize=8)
ax3.set_xlabel('Carrier frequency (Hz)', fontsize=8)
ax3.set_ylabel('Correlation', fontsize=8)
ax3.set_title(f'Correlation between orthogonalized '
              f'spontaneous signals from Visual ' 
              f'sensory areas', fontsize=8)
ax3.legend(fontsize=8)
ax3.grid(False)


plt.savefig('/home/senthilp/Desktop/new_plot_check.png', dpi=600)