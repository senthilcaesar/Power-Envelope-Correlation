import numpy as np
import nibabel as nib
from scipy.io import loadmat
from regfusion import vol_to_fsaverage

name = 'cluster'
fname = f'Graph_{name}.mat'
corr = loadmat(fname)
sub_dir = '/Users/senthilp/Desktop/Wen/degree/correlation'
input_path = '/Applications/freesurfer/7.1.1/subjects/fsaverage/mri/brain.mgz'
atlas = '/Users/senthilp/Desktop/Wen/degree/atlas/shen_freesurferSpace.nii.gz'

freq1 = corr[name][0]
freq2 = corr[name][1]
freq3 = corr[name][2]
freq4 = corr[name][3]
freq5 = corr[name][4]
freq6 = corr[name][5]

print(freq1.max(), freq2.max(), freq3.max(), freq4.max(), freq5.max(), freq6.max())
print(freq1.min(), freq2.min(), freq3.min(), freq4.min(), freq5.min(), freq6.min())

freq_lst = [freq1, freq2, freq3, freq4, freq5, freq6]

for k, freq in enumerate(freq_lst):
    print(f'Processing freq{k+1}...')
    obj = nib.load(atlas)
    data = nib.load(atlas).get_fdata()
    want = np.arange(1,269)
    map_data = np.zeros((data.shape))
    
    for i, id in enumerate(want):
        index = np.where(data == id)
        map_data[index] = freq[i]
    
    output_file = f'{sub_dir}/{name}_mapped_array_freq{k+1}.nii.gz'
    t1 = nib.load(input_path)
    affine = t1.affine
    hdr = t1.header
    result_img = nib.Nifti1Image(map_data, affine, header=hdr)
    result_img.to_filename(output_file)
    
    # Map volume to surface
    lh, rh = vol_to_fsaverage(output_file, f'{name}_mapped_vol_freq{k+1}')
    del output_file, t1, affine, hdr, result_img, map_data, freq, index, data, obj