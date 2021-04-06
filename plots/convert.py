import nibabel as nib
import numpy as np

convert = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
age_arr = ['18to29', '30to39', '40to49', '50to59', '60to69', '70to79', '80to88']
seed_arr = ['scLeft', 'acLeft' ,'vcLeft', 'mtLeft', 'mtlLeft', 'smcLeft', 'lpcLeft', 'lpcLeft', 'dpfcLeft', 'tmpcLeft',
            'scRight', 'acRight', 'vcRight', 'mtRight', 'mtlRight', 'smcRight', 'lpcRight', 'dpfcRight', 'tmpcRight',
            'mpfc', 'sma']
for age in age_arr:
    print('Age : ', age)
    for freq in convert:    
        freq = str(freq)
        print('Frequency : ', freq)
        for seed in seed_arr:
            print('Seed : ', seed)
            vol_dir = f'/Users/senthilp/Desktop/average/{age}_avg'
            vol_fname = f'{vol_dir}/{age}_true_{freq}_{seed}.nii.gz'
            t1_fname = 'T1.mgz'

            t1 = nib.load(t1_fname)
            img = nib.load(vol_fname)
            imgF32 = img.get_fdata()

            print('Before : ', imgF32.max())
            imgF32[ imgF32 >= 0.1 ] = 0.1

            affine = t1.affine
            hdr = t1.header
            result_img = nib.Nifti1Image(imgF32, affine, header=hdr)
            result_img.to_filename(vol_fname)

            check = nib.load(vol_fname).get_fdata()
            print('After : ', check.max())
