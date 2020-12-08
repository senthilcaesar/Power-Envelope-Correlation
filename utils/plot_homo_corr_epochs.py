import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

cases = '/home/senthilp/caesar/camcan/cc700/freesurfer_output/18to30.txt'
subjects_dir = '/home/senthilp/caesar/camcan/cc700/freesurfer_output'
flag = 'true'
with open(cases) as f:
     case_list = f.read().splitlines()


freq = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
y_corr = [0.1, 0.2, 0.3, 0.4, 0.5]
spacing=30
sensory_mean = {'scLeft':None, 'acLeft':None, 'vcLeft':None,
                'scRight':None, 'acRight':None, 'vcRight':None}

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 5))

x_pnts = np.arange(len(freq))

mean_dict = []
for label in sensory_mean:
    myDict = {key: [] for key in freq}
    for subject in case_list:
        DATA_DIR = Path(f'{subjects_dir}', f'{subject}', 'mne_files')
        for val in freq:
            corr_data_file = f'{subjects_dir}/{subject}/mne_files/{subject}_'\
                            f'corr_ortho_{flag}_{spacing}_{val}_{label}_wholebrain_epoch.npy'
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
    

corr_sc = [(a + b)/2 for a, b in zip(sensory_mean['scLeft'], sensory_mean['scRight'])]
corr_ac = [(a + b)/2 for a, b in zip(sensory_mean['acLeft'], sensory_mean['acRight'])]
corr_vc = [(a + b)/2 for a, b in zip(sensory_mean['vcLeft'], sensory_mean['vcRight'])]



cases2 = '/home/senthilp/caesar/camcan/cc700/freesurfer_output/42to58.txt'
subjects_dir2 = '/home/senthilp/caesar/camcan/cc700/freesurfer_output'
flag2 = 'true'
with open(cases2) as f2:
     case_list2 = f2.read().splitlines()


freq2 = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
y_corr2 = [0.1, 0.2, 0.3, 0.4, 0.5]
spacing2=30
sensory_mean2 = {'scLeft':None, 'acLeft':None, 'vcLeft':None,
                'scRight':None, 'acRight':None, 'vcRight':None}

mean_dict2 = []
for label2 in sensory_mean2:
    myDict2 = {key2: [] for key2 in freq2}
    for subject2 in case_list2:
        DATA_DIR2 = Path(f'{subjects_dir}', f'{subject2}', 'mne_files')
        for val2 in freq2:
            corr_data_file2 = f'{subjects_dir}/{subject2}/mne_files/{subject2}_'\
                            f'corr_ortho_{flag2}_{spacing2}_{val2}_{label2}_wholebrain_epoch.npy'
            if Path(corr_data_file2).exists():
                if label2 == 'scLeft':
                    corr_data2 = float(np.load(corr_data_file2)[1]) * 1.73
                elif label2 == 'acLeft':
                    corr_data2 = float(np.load(corr_data_file2)[3]) * 1.73
                elif label2 == 'vcLeft':
                    corr_data2 = float(np.load(corr_data_file2)[5]) * 1.73   
                elif label2 == 'scRight':
                    corr_data2 = float(np.load(corr_data_file2)[0]) * 1.73
                elif label2 == 'acRight':
                    corr_data2 = float(np.load(corr_data_file2)[2]) * 1.73
                elif label2 == 'vcRight':
                    corr_data2 = float(np.load(corr_data_file2)[4]) * 1.73   
                myDict2[val2].append(corr_data2)
    mean_dict2.append(myDict2)
    sensory_mean2[label2] = [np.mean(value2) for key2, value2 in myDict2.items()]
    

corr_sc2 = [(a + b)/2 for a, b in zip(sensory_mean2['scLeft'], sensory_mean2['scRight'])]
corr_ac2 = [(a + b)/2 for a, b in zip(sensory_mean2['acLeft'], sensory_mean2['acRight'])]
corr_vc2 = [(a + b)/2 for a, b in zip(sensory_mean2['vcLeft'], sensory_mean2['vcRight'])]


ax1.plot(x_pnts, corr_ac, '-ok', color='red', markerfacecolor='black', 
        label='Auditory young', alpha=1, markersize=3, linewidth=0.5)
ax1.plot(x_pnts, corr_ac2, '-ok', color='green', markerfacecolor='black', 
        label='Auditory middle', alpha=1, markersize=3, linewidth=0.5)

ax2.plot(x_pnts, corr_sc, '-ok', color='red', markerfacecolor='black', 
        label='Somat young', alpha=1, markersize=3, linewidth=0.5)
ax2.plot(x_pnts, corr_sc2, '-ok', color='green', markerfacecolor='black', 
        label='Somat middle', alpha=1, markersize=3, linewidth=0.5)

ax3.plot(x_pnts, corr_vc, '-ok', color='red', markerfacecolor='black', 
        label='Visual young', alpha=1, markersize=3, linewidth=0.5)
ax3.plot(x_pnts, corr_vc2, '-ok', color='green', markerfacecolor='black', 
        label='Visual middle', alpha=1, markersize=3, linewidth=0.5)



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

fig.suptitle('Segmented data', fontsize=20)
plt.savefig('/home/senthilp/Desktop/test_epoch.png', dpi=600)