import numpy as np
import mne
from pathlib import Path

cases = '/home/senthilp/caesar/camcan/cc700/freesurfer_output/18to29.txt'
subjects_dir = '/home/senthilp/caesar/camcan/cc700/freesurfer_output'
flag = 'true'
with open(cases) as f:
     case_list = f.read().splitlines()


freq = [6, 8, 12, 16]
spacing=7.8
sensory_mean = {'lpcLeft':None, 'dpfcLeft':None, 'tmpcLeft':None, 'mtLeft':None, 'lpcRight':None, 'dpfcRight':None, 'tmpcRight':None, 'mtRight':None}

for label in sensory_mean:
    myDict = {key: [] for key in freq}
    for subject in case_list:
        DATA_DIR = Path(f'{subjects_dir}', f'{subject}', 'mne_files')
        for val in freq:
            corr_data_file = f'scp -pr senthilp@128.197.62.176:{subjects_dir}/{subject}/mne_files/{subject}_'\
                            f'corr_ortho_{flag}_{spacing}_{val}_{label}.npy .'
            print(corr_data_file)
