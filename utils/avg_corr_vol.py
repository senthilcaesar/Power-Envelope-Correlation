import numpy as np
from pathlib import Path
import nibabel as nib


cases = '/home/senthil/caesar/camcan/cc700/freesurfer_output/50.txt'
subjects_dir = '/home/senthil/caesar/camcan/cc700/freesurfer_output'
flag = 'true'
sensor = ['sc', 'ac', 'vc']
freqs = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
with open(cases) as f:
      case_list = f.read().splitlines()

t1_fname = '/home/senthil/caesar/camcan/cc700/freesurfer_output/fsaverage/mri/brain.mgz'
t1 = nib.load(t1_fname)
affine = t1.affine
hdr = t1.header

for freq in freqs:    
    for label in sensor:
        subjs_corr = np.zeros([256,256,256])
        for subject in case_list:
            DATA_DIR = Path(f'{subjects_dir}', f'{subject}', 'mne_files')
            corr_data_warp = f'{DATA_DIR}/{subject}_{freq}_{label}_antsWarped.nii.gz'
            data_npy = nib.load(corr_data_warp).get_fdata()
            np.add(subjs_corr, data_npy, out=subjs_corr)
        np.divide(subjs_corr, len(case_list), out=subjs_corr)
        np.multiply(subjs_corr, 1.73, out=subjs_corr)
    
        output = f'{subjects_dir}/average/50_{flag}_{freq}_{label}.nii.gz'
        result_img = nib.Nifti1Image(subjs_corr, affine, header=hdr)
        result_img.to_filename(output)
        print(output)
        print(subjs_corr.max())
        print('\n')

