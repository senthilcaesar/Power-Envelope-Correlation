import pickle
import mne
import os
import nibabel
import numpy as np
from mne.transforms import apply_trans
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import nibabel as nib
from nilearn import plotting


subjects_dir = '/home/senthil/Downloads/tmp'
subject = 'sub-CC221373'
src_space_fname = '/home/senthil/Downloads/tmp/sub-CC221373/mne_files/sub-CC221373-src.fif.gz'
src_space = mne.read_source_spaces(src_space_fname)
lh_surf_coord = src_space[0]['rr']     # Triangle Mesh coordinates
lh_triangle_idx = src_space[0]['tris'] # traingular mesh face of 3 vertices

'''
Source space is in Freesurfer MRI coordinates
'''

t1_fname = os.path.join(subjects_dir, subject, 'mri', 'T1.mgz')
t1 = nib.load(t1_fname)
data = np.asarray(t1.dataobj)

vox_mri_t = t1.header.get_vox2ras_tkr()
mri_vox_t = np.linalg.inv(vox_mri_t)

sources = []
for src_ in src_space:
    points = src_['rr'][src_['inuse'].astype(bool)]
    sources.append(apply_trans(mri_vox_t, points * 1e3))
    sources = np.concatenate(sources, axis=0)
sources = np.round(sources)


m = '/home/senthil/Downloads/tmp/sub-CC221373/mne_files/coords.pkl'
f = open(m, 'rb')
source_mne = pickle.load(f)[0]

img = np.zeros([256,256,256])

corr_flag = 'true'
marker_name = f'/home/senthil/Downloads/tmp/marker_{corr_flag}'
stat_img = 'test_pix.nii.gz'
display = plotting.plot_anat(t1_fname,draw_cross=False,
                  title="plot_roi", cut_coords=[24, 17, 16])
coords = [(24, 17, 16)]
display.add_markers(coords, marker_color='y', marker_size=5)
display.savefig(marker_name, dpi=500)

for i, val in enumerate(sources):
    if (val == np.array([105,  93, 126])).all():
        print(i)
        

a = np.load(f'/home/senthil/Downloads/tmp/corr_ortho_{corr_flag}.npy')
b = a[80040,:]

for idx, val in enumerate(sources):
    i, j, k = int(val[0]), int(val[1]), int(val[2])
    img[i][j][k] = b[idx]

affine = t1.affine
hdr = t1.header
result_img = nib.Nifti1Image(img, affine, header=hdr)
result_img.to_filename('test_pix.nii.gz')

output_html = f'/home/senthil/Downloads/tmp/corr_{corr_flag}.html'
html_view = plotting.view_img(stat_img, bg_img=t1_fname, 
                          threshold='0%',
                          resampling_interpolation='nearest',
                          cmap='RdYlGn')
html_view.save_as_html(output_html)

'''
  vertex indices ( every index value for vertices will
  select a coordinate from lh_surf_coord )
 
  lh_data source data is mapped to lh_ver_idx which is then mapped to lh_surf_coord
 
  Each triangle lh_triangle_idx consist of 3 vertices
 
'''




