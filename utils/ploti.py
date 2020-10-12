import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch

cases = '/home/senthil/caesar/camcan/cc700/freesurfer_output/50.txt'
subjects_dir = '/home/senthil/caesar/camcan/cc700/freesurfer_output'
flag = 'True'
with open(cases) as f:
     case_list = f.read().splitlines()


corr_2 = []
corr_4 = []
corr_8 = []
corr_16 = []
corr_32 = []
corr_64 = []
corr_128 = []
for case in case_list:
    subject = case
    DATA_DIR = Path(f'{subjects_dir}', f'{subject}', 'mne_files')
    corr2_data = f'{DATA_DIR}/{subject}_corr_ortho_true_5_2.npy'
    corr4_data = f'{DATA_DIR}/{subject}_corr_ortho_true_5_4.npy'
    corr8_data = f'{DATA_DIR}/{subject}_corr_ortho_true_5_8.npy'
    corr16_data = f'{DATA_DIR}/{subject}_corr_ortho_true_5_16.npy'
    corr32_data = f'{DATA_DIR}/{subject}_corr_ortho_true_5_32.npy'
    corr64_data = f'{DATA_DIR}/{subject}_corr_ortho_true_5_64.npy'
    corr128_data = f'{DATA_DIR}/{subject}_corr_ortho_true_5_128.npy'
    corr2 = float(np.load(corr2_data))
    corr4 = float(np.load(corr4_data))
    corr8 = float(np.load(corr8_data))
    corr16 = float(np.load(corr16_data))
    corr32 = float(np.load(corr32_data))
    corr64 = float(np.load(corr64_data))
    corr128 = float(np.load(corr128_data))
    corr_2.append(corr2)
    corr_4.append(corr4)
    corr_8.append(corr8)
    corr_16.append(corr16)
    corr_32.append(corr32)
    corr_64.append(corr64)
    corr_128.append(corr128)
    
    

a = np.arange(7)
b = np.arange(5)
x = [2, 4, 8, 16, 32, 64, 128]
y = [0, 0.04, 0.08, 0.12, 0.16]

corr2_avg = np.mean(corr_2)
corr4_avg = np.mean(corr_4)
corr8_avg = np.mean(corr_8)
corr16_avg = np.mean(corr_16)
corr32_avg = np.mean(corr_32)
corr64_avg = np.mean(corr_64)
corr128_avg = np.mean(corr_128)

fig, ax = plt.subplots(figsize=(5, 3))

ax.errorbar(np.full((50), 0), corr_2, fmt='gold')
ax.errorbar(np.full((50), 1), corr_4, fmt='gold')
ax.errorbar(np.full((50), 2), corr_8, fmt='gold')
ax.errorbar(np.full((50), 3), corr_16, fmt='gold')
ax.errorbar(np.full((50), 4), corr_32, fmt='gold')
ax.errorbar(np.full((50), 5), corr_64, fmt='gold')
ax.errorbar(np.full((50), 6), corr_128, fmt='gold')

ax.plot(0,corr2_avg, marker='x', color='k', markersize=4)
ax.plot(1,corr4_avg, marker='x', color='k', markersize=4)
ax.plot(2,corr8_avg, marker='x', color='k', markersize=4)
ax.plot(3,corr16_avg, marker='x', color='k', markersize=4)
ax.plot(4,corr32_avg, marker='x', color='k', markersize=4)
ax.plot(5,corr64_avg, marker='x', color='k', markersize=4)
ax.plot(6,corr128_avg, marker='x', color='k', markersize=4)

xyA = (0.0, corr2_avg)
xyB = (1.0, corr4_avg)
xyC = (2.0, corr8_avg)
xyD = (3.0, corr16_avg)
xyE = (4.0, corr32_avg)
xyF = (5.0, corr64_avg)
xyG = (6.0, corr128_avg)

gold_patch = mpatches.Patch(color='gold', label='Somat', lw=0)
red_patch = mpatches.Patch(color='red', label='Auditory', lw=0)
blue_patch = mpatches.Patch(color='blue', label='Visual', lw=0)

con1 = ConnectionPatch(xyA, xyB, coordsA='data')
con2 = ConnectionPatch(xyB, xyC, coordsA='data')
con3 = ConnectionPatch(xyC, xyD, coordsA='data')
con4 = ConnectionPatch(xyD, xyE, coordsA='data')
con5 = ConnectionPatch(xyE, xyF, coordsA='data')
con6 = ConnectionPatch(xyF, xyG, coordsA='data')
ax.add_artist(con1)
ax.add_artist(con2)
ax.add_artist(con3)
ax.add_artist(con4)
ax.add_artist(con5)
ax.add_artist(con6)

ax.set_xticks(a) 
ax.set_xticklabels(x)
ax.set_yticks(y) 
ax.set_xlabel('Carrier frequency (Hz)')
ax.set_ylabel('Correlation')
ax.legend(handles=[red_patch, gold_patch, blue_patch], loc='upper right', fontsize=8)


