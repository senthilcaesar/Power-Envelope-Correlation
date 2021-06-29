import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def compute_sem(mean_dict):
    from scipy import stats
    work = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[],
    		6:[], 7:[], 8:[], 9:[], 10:[], 11:[],
    		12:[], 13:[], 14:[], 15:[], 16:[], 17:[]}

    for i, roi in enumerate(mean_dict):
        for each_freq in roi:
            work[i].append(stats.sem(roi[each_freq]))
        
    sem_sc = [(a + b)/2 for a, b in zip(work[0], work[9])]
    sem_ac = [(a + b)/2 for a, b in zip(work[1], work[10])]
    sem_vc = [(a + b)/2 for a, b in zip(work[2], work[11])]
    sem_mt = [(a + b)/2 for a, b in zip(work[3], work[12])]
    sem_mtl = [(a + b)/2 for a, b in zip(work[4], work[13])]
    sem_smc = [(a + b)/2 for a, b in zip(work[5], work[14])]
    sem_lpc = [(a + b)/2 for a, b in zip(work[6], work[15])]
    sem_dpfc = [(a + b)/2 for a, b in zip(work[7], work[16])]
    sem_tmpc = [(a + b)/2 for a, b in zip(work[8], work[17])]
    
    return sem_sc, sem_ac, sem_vc, sem_mt, sem_mtl, sem_smc, sem_lpc, sem_dpfc, sem_tmpc


freq = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
y_corr = [0, 0.04, 0.08, 0.12, 0.16]
spacing=30
flag = 'true'
sensory_mean = {'scLeft':None, 'acLeft':None, 'vcLeft':None, 'mtLeft': None, 'mtlLeft': None, 'smcLeft': None, 'lpcLeft': None, 'dpfcLeft': None, 'tmpcLeft': None,
                'scRight':None, 'acRight':None, 'vcRight':None, 'mtRight': None, 'mtlRight': None, 'smcRight': None, 'lpcRight': None, 'dpfcRight': None, 'tmpcRight': None}

fig, ax = plt.subplots(3, 3, figsize=(30, 15))
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
                                f'corr_ortho_{flag}_{spacing}_{val}_{label}.npy'
                if Path(corr_data_file).exists():
                    if label == 'scLeft':
                        corr_data = float(np.load(corr_data_file)[1]) * 1.73
                    elif label == 'acLeft':
                        corr_data = float(np.load(corr_data_file)[3]) * 1.73
                    elif label == 'vcLeft':
                        corr_data = float(np.load(corr_data_file)[5]) * 1.73
                    elif label == 'mtLeft':
                        corr_data = float(np.load(corr_data_file)[7]) * 1.73
                    elif label == 'mtlLeft':
                        corr_data = float(np.load(corr_data_file)[9]) * 1.73
                    elif label == 'smcLeft':
                        corr_data = float(np.load(corr_data_file)[11]) * 1.73
                    elif label == 'lpcLeft':
                        corr_data = float(np.load(corr_data_file)[13]) * 1.73
                    elif label == 'dpfcLeft':
                        corr_data = float(np.load(corr_data_file)[15]) * 1.73
                    elif label == 'tmpcLeft':
                        corr_data = float(np.load(corr_data_file)[17]) * 1.73

                    elif label == 'scRight':
                        corr_data = float(np.load(corr_data_file)[0]) * 1.73
                    elif label == 'acRight':
                        corr_data = float(np.load(corr_data_file)[2]) * 1.73
                    elif label == 'vcRight':
                        corr_data = float(np.load(corr_data_file)[4]) * 1.73
                    elif label == 'mtRight':
                        corr_data = float(np.load(corr_data_file)[6]) * 1.73
                    elif label == 'mtlRight':
                        corr_data = float(np.load(corr_data_file)[8]) * 1.73
                    elif label == 'smcRight':
                        corr_data = float(np.load(corr_data_file)[10]) * 1.73
                    elif label == 'lpcRight':
                        corr_data = float(np.load(corr_data_file)[12]) * 1.73
                    elif label == 'dpfcRight':
                        corr_data = float(np.load(corr_data_file)[14]) * 1.73
                    elif label == 'tmpcRight':
                        corr_data = float(np.load(corr_data_file)[16]) * 1.73

                    myDict[val].append(corr_data)
        mean_dict.append(myDict)
        sensory_mean[label] = [np.mean(value) for key, value in myDict.items()]
    
    mean_corr_sc = [(a + b)/2 for a, b in zip(sensory_mean['scLeft'], sensory_mean['scRight'])]
    mean_corr_ac = [(a + b)/2 for a, b in zip(sensory_mean['acLeft'], sensory_mean['acRight'])]
    mean_corr_vc = [(a + b)/2 for a, b in zip(sensory_mean['vcLeft'], sensory_mean['vcRight'])]
    mean_corr_mt = [(a + b)/2 for a, b in zip(sensory_mean['mtLeft'], sensory_mean['mtRight'])]
    mean_corr_mtl = [(a + b)/2 for a, b in zip(sensory_mean['mtlLeft'], sensory_mean['mtlRight'])]
    mean_corr_smc = [(a + b)/2 for a, b in zip(sensory_mean['smcLeft'], sensory_mean['smcRight'])]
    mean_corr_lpc = [(a + b)/2 for a, b in zip(sensory_mean['lpcLeft'], sensory_mean['lpcRight'])]
    mean_corr_dpfc = [(a + b)/2 for a, b in zip(sensory_mean['dpfcLeft'], sensory_mean['dpfcRight'])]
    mean_corr_tmpc = [(a + b)/2 for a, b in zip(sensory_mean['tmpcLeft'], sensory_mean['tmpcRight'])]
    
    sem_sc, sem_ac, sem_vc, sem_mt, sem_mtl, sem_smc, sem_lpc, sem_dpfc, sem_tmpc = compute_sem(mean_dict)

    plus_sc = [(a + b) for a, b in zip(mean_corr_sc, sem_sc)]
    plus_ac = [(a + b) for a, b in zip(mean_corr_ac, sem_ac)]
    plus_vc = [(a + b) for a, b in zip(mean_corr_vc, sem_vc)]
    plus_mt = [(a + b) for a, b in zip(mean_corr_mt, sem_mt)]
    plus_mtl = [(a + b) for a, b in zip(mean_corr_mtl, sem_mtl)]
    plus_smc = [(a + b) for a, b in zip(mean_corr_smc, sem_smc)]
    plus_lpc = [(a + b) for a, b in zip(mean_corr_lpc, sem_lpc)]
    plus_dpfc = [(a + b) for a, b in zip(mean_corr_dpfc, sem_dpfc)]
    plus_tmpc = [(a + b) for a, b in zip(mean_corr_tmpc, sem_tmpc)]

    minus_sc = [(a - b) for a, b in zip(mean_corr_sc, sem_sc)]
    minus_ac = [(a - b) for a, b in zip(mean_corr_ac, sem_ac)]
    minus_vc = [(a - b) for a, b in zip(mean_corr_vc, sem_vc)]
    minus_mt = [(a - b) for a, b in zip(mean_corr_mt, sem_mt)]
    minus_mtl = [(a - b) for a, b in zip(mean_corr_mtl, sem_mtl)]
    minus_smc = [(a - b) for a, b in zip(mean_corr_smc, sem_smc)]
    minus_lpc = [(a - b) for a, b in zip(mean_corr_lpc, sem_lpc)]
    minus_dpfc = [(a - b) for a, b in zip(mean_corr_dpfc, sem_dpfc)]
    minus_tmpc = [(a - b) for a, b in zip(mean_corr_tmpc, sem_tmpc)]
    
    ax[0][0].plot(x_pnts, mean_corr_ac, '-ok', color=color_plot[i], markerfacecolor='black', 
        label=f'{age_file}', alpha=1, markersize=3, linewidth=0.5)
    ax[0][0].fill_between(x_pnts, plus_ac, minus_ac, color=color_plot[i], alpha=0.05)
    
    ax[0][1].plot(x_pnts, mean_corr_sc, '-ok', color=color_plot[i], markerfacecolor='black', 
        label=f'{age_file}', alpha=1, markersize=3, linewidth=0.5)
    ax[0][1].fill_between(x_pnts, plus_sc, minus_sc, color=color_plot[i], alpha=0.05)
    
    ax[0][2].plot(x_pnts, mean_corr_vc, '-ok', color=color_plot[i], markerfacecolor='black', 
        label=f'{age_file}', alpha=1, markersize=3, linewidth=0.5)
    ax[0][2].fill_between(x_pnts, plus_vc, minus_vc, color=color_plot[i], alpha=0.05)

    ax[1][0].plot(x_pnts, mean_corr_mt, '-ok', color=color_plot[i], markerfacecolor='black', 
        label=f'{age_file}', alpha=1, markersize=3, linewidth=0.5)
    ax[1][0].fill_between(x_pnts, plus_mt, minus_mt, color=color_plot[i], alpha=0.05)

    ax[1][1].plot(x_pnts, mean_corr_mtl, '-ok', color=color_plot[i], markerfacecolor='black', 
        label=f'{age_file}', alpha=1, markersize=3, linewidth=0.5)
    ax[1][1].fill_between(x_pnts, plus_mtl, minus_mtl, color=color_plot[i], alpha=0.05)

    ax[1][2].plot(x_pnts, mean_corr_smc, '-ok', color=color_plot[i], markerfacecolor='black', 
        label=f'{age_file}', alpha=1, markersize=3, linewidth=0.5)
    ax[1][2].fill_between(x_pnts, plus_smc, minus_smc, color=color_plot[i], alpha=0.05)

    ax[2][0].plot(x_pnts, mean_corr_lpc, '-ok', color=color_plot[i], markerfacecolor='black', 
        label=f'{age_file}', alpha=1, markersize=3, linewidth=0.5)
    ax[2][0].fill_between(x_pnts, plus_lpc, minus_lpc, color=color_plot[i], alpha=0.05)

    ax[2][1].plot(x_pnts, mean_corr_dpfc, '-ok', color=color_plot[i], markerfacecolor='black', 
        label=f'{age_file}', alpha=1, markersize=3, linewidth=0.5)
    ax[2][1].fill_between(x_pnts, plus_dpfc, minus_dpfc, color=color_plot[i], alpha=0.05)

    ax[2][2].plot(x_pnts, mean_corr_tmpc, '-ok', color=color_plot[i], markerfacecolor='black', 
        label=f'{age_file}', alpha=1, markersize=3, linewidth=0.5)
    ax[2][2].fill_between(x_pnts, plus_tmpc, minus_tmpc, color=color_plot[i], alpha=0.05)


rois = ['Auditory', 'Somatosensory', 'Visual', 'Middle temporal area',
        'Medial temporal lobe', 'Sensorimotor cortex', 'LPC', 'DPFC', 'TMPC']

for i, ax in enumerate(fig.axes):
	ax.set_xticks(x_pnts)
	ax.set_yticks(y_corr)
	ax.set_xticklabels(freq, fontsize=8)
	ax.set_yticklabels(y_corr, fontsize=8)
	ax.set_xlabel('Carrier frequency (Hz)', fontsize=8)
	ax.set_ylabel('Correlation', fontsize=8)
	ax.set_title(f'Correlation between orthogonalized '
	              f'spontaneous signals from {rois[i]} ' 
	              f'sensory areas', fontsize=8)
	ax.legend(fontsize=8)
	ax.grid(False)


plt.savefig('/home/senthilp/Desktop/new_plot.png', dpi=800)
