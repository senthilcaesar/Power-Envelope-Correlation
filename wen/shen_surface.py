import pandas as pd
import numpy as np
import nibabel as nib
from regfusion import vol_to_fsaverage

filename = '/Users/senthilp/Desktop/wen/degree/shen268_subnetwork10.csv'
df = pd.read_csv(filename)
df['color'] = pd.Categorical(df['label'].astype(str)).codes
g = df.groupby('category')['node'].apply(lambda x: list(np.unique(x)))
h = df.groupby('category')['label'].apply(lambda x: list(np.unique(x)))

input_path = '/Applications/freesurfer/7.1.1/subjects/fsaverage/mri/brain.mgz'
atlas = '/Users/senthilp/Desktop/wen/degree/atlas/shen_freesurferSpace.nii.gz'
obj = nib.load(atlas)
data = nib.load(atlas).get_fdata()
map_data = np.zeros((data.shape))

want = np.arange(1,269)
want_new = np.zeros((269))

for i, idx in enumerate(g):
    want_new[np.array(idx)-1] = i

for j, id in enumerate(want):
    index = np.where(data == id)
    map_data[index] = want_new[j]+1
    
# Create nifti brain image volume
output_file = 'shen_mapped_array.nii.gz'
t1 = nib.load(input_path)
affine = t1.affine
hdr = t1.header
hdr = hdr.set_data_dtype(np.uint8)
result_img = nib.Nifti1Image(map_data, affine, header=hdr)
result_img.to_filename(output_file)


# Map volume to surface
lh, rh = vol_to_fsaverage(output_file, out_dir='shen_surf', interp='nearest')



import seaborn as sns
import matplotlib.pyplot as plt
gyr = ['#4682b4','#f5f5f5', '#cd3e4e', '#781286', '#f27efa',
       '#46f2f4', '#dcf8a4', '#e69422', '#fcff2b', '#00760e']
sns.palplot(sns.color_palette(gyr))
plt.savefig('/Users/senthilp/Desktop/color.png', dpi=600)



from nilearn import datasets
a = datasets.fetch_coords_power_2011()
roi = a['rois']
arr = pd.DataFrame(roi).to_numpy()











