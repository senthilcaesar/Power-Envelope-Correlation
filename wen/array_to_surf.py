from genericpath import exists
import nibabel as nib
import numpy as np
from regfusion import vol_to_fsaverage
import nibabel.processing
import os.path
from os import path
import scipy.io


sub_dir = '/Users/senthilp/Desktop/Wen/degree/npy80'

input_path = '/Applications/freesurfer/7.1.1/subjects/fsaverage/mri/brain.mgz'

# Register shen atlas to freesurfer fsaverage T1
atlas = '/Users/senthilp/Desktop/Wen/degree/atlas/shen_freesurferSpace.nii.gz'

# 268 array values
degree_avg = np.load(f'{sub_dir}/degree_avg.npy')

# Load shen atlas data
obj = nib.load(atlas)
data = nib.load(atlas).get_fdata()

# Uniques atlas ID's from 1 to 268
want = np.arange(1,269)

# Create new matrix
map_data = np.zeros((data.shape))

# Loop through the want ID's and copy corr values to the matrix
for i, id in enumerate(want):
    index = np.where(data == id)
    map_data[index] = degree_avg[i]


# Create nifti brain image volume
output_file = f'{sub_dir}/degree_mapped_array.nii.gz'
t1 = nib.load(input_path)
affine = t1.affine
hdr = t1.header
result_img = nib.Nifti1Image(map_data, affine, header=hdr)
result_img.to_filename(output_file)

# Map volume to surface
lh, rh = vol_to_fsaverage(output_file, 'mapped_vol_80')
