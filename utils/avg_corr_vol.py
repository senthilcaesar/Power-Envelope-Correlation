import numpy as np
from pathlib import Path
import nibabel as nib
import os
from settings import Settings
import os.path


if __name__ == '__main__':

    settings = Settings()

    data_params = settings['DATA']
    hyper_params = settings['PARAMS']
    common_params = settings['COMMON']

    cases = data_params['cases']
    subjects_dir = common_params['subjects_dir']
    spacing = hyper_params['vol_spacing']
    freqs = hyper_params['freqs']
    flag = hyper_params['ortho_flag']
    sensor = hyper_params['sensor']
    t1_fname = data_params['t1_fsaverage']

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
                corr_data_warp = f'{subjects_dir}/{subject}/mne_files/{subject}_{flag}_7.8_{freq}_corr_{label}_antsWarped.nii.gz'
                if os.path.isfile(corr_data_warp):
                    count += 1
                    data_npy = nib.load(corr_data_warp).get_fdata()
                    np.add(subjs_corr, data_npy, out=subjs_corr)
            np.divide(subjs_corr, count, out=subjs_corr)
            #np.multiply(subjs_corr, 1.73, out=subjs_corr)
            output = f'{subjects_dir}/average/18to29_avg/{case_name}_{flag}_{freq}_{label}.nii.gz'
            result_img = nib.Nifti1Image(subjs_corr, affine, header=hdr)
            result_img.to_filename(output)
            print(output)
            print(subjs_corr.max())
            print('\n')
