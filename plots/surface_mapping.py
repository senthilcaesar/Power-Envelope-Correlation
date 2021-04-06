from regfusion import vol_to_fsaverage
import nibabel as nib
import numpy as np

age = ['18to29','30to39','40to49','50to59','60to69','70to79','80to88']
freqs = [6, 8, 12, 16]
seed='mtRight'

for val in age:
    for freq in freqs:
        print(f'Mapping volume to surface for {val} {freq}')
        input_vol=f'seedVol/{val}_true_{freq}_{seed}.nii.gz'
        print(input_vol)
        vol_data = nib.load(input_vol).get_fdata()
        print(vol_data.min(), vol_data.max())
        lh, rh = vol_to_fsaverage(input_vol, 'output')
