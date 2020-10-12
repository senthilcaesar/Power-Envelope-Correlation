import numpy as np
from pathlib import Path
import nibabel as nib


cases = '/home/senthil/caesar/camcan/cc700/freesurfer_output/50.txt'
subjects_dir = '/home/senthil/caesar/camcan/cc700/freesurfer_output'
flag = 'True'
with open(cases) as f:
     case_list = f.read().splitlines()
     
subjs_cov = np.zeros([256,256,256])
for i, case in enumerate(case_list):
    subject = case
    DATA_DIR = Path(f'{subjects_dir}', f'{subject}', 'mne_files')
    corr_data_warp = f'{DATA_DIR}/{subject}_ants_ortho{flag}_epochWarped.nii.gz'
    subjs_cov = subjs_cov + nib.load(corr_data_warp).get_fdata()
    print(f'Case completed {i}...')
subjs_cov = subjs_cov / len(case_list)

t1_fname = '/home/senthil/mne_data/MNE-sample-data/subjects/fsaverage/mri/brain.mgz'
t1 = nib.load(t1_fname)
affine = t1.affine
hdr = t1.header
result_img = nib.Nifti1Image(subjs_cov, affine, header=hdr)
result_img.to_filename(f'/home/senthil/Downloads/50_{flag}_epoch.nii.gz')


