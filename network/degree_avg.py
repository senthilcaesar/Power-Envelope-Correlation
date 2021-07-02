import numpy as np
from pathlib import Path
import nibabel as nib
import os
import os.path


if __name__ == '__main__':

    cases = '/home/senthilp/caesar/camcan/cc700/freesurfer_output/80to88.txt'
    subjects_dir = '/home/senthilp/caesar/camcan/cc700/freesurfer_output'
    spacing = '7.8'
    freqs = [12,]
    sensor = ['degreeMapped',]
    t1_fname = '/home/senthilp/freesurfer/subjects/fsaverage/mri/brain.mgz'

    with open(cases) as f:
        case_list = f.read().splitlines()

    t1 = nib.load(t1_fname)
    affine = t1.affine
    hdr = t1.header

    case_name = os.path.splitext(os.path.basename(cases))[0]

    for freq in freqs:    
        for label in sensor:
            subjs_corr = np.zeros([256,256,256])
            count = 0
            for i, subject in enumerate(case_list):
                print(subject, ' ' , i)
                DATA_DIR = Path(f'{subjects_dir}', f'{subject}', 'mne_files')
                corr_data_warp = f'{subjects_dir}/{subject}/mne_files/{subject}_{label}_{spacing}_{freq}_antsWarped.nii.gz'
                if os.path.isfile(corr_data_warp):
                    count += 1
                    data_npy = nib.load(corr_data_warp).get_fdata()
                    np.add(subjs_corr, data_npy, out=subjs_corr)
            np.divide(subjs_corr, count, out=subjs_corr)
            output = f'{subjects_dir}/average/{case_name}_{label}_{spacing}_{freq}.nii.gz'
            result_img = nib.Nifti1Image(subjs_corr, affine, header=hdr)
            result_img.to_filename(output)
            print(output)
            print(subjs_corr.max())
            print('\n')
