import pickle
import mne
import os
import nibabel
import numpy as np
from mne.transforms import apply_trans
import nibabel as nib
from nilearn import plotting
from scipy.signal import convolve2d


def eight_neighbor_average_convolve2d(x):
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0

    neighbor_sum = convolve2d(
        x, kernel, mode='same',
        boundary='fill', fillvalue=0)

    num_neighbor = convolve2d(
        np.ones(x.shape), kernel, mode='same',
        boundary='fill', fillvalue=0)

    return neighbor_sum / num_neighbor


def test_cv(img):
    img[img > 0.0] = 1
    img = np.swapaxes(img,0,1) # coronal
    img = np.swapaxes(img,0,2)
    import cv2
    for i in range(0,len(img)):
        cv2.imwrite("/home/senthil/Desktop/pics/"+str(i)+".png",img[i]*255.0)
        
        
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

corr_flag = 'false'
marker_name = f'/home/senthil/Downloads/tmp/marker_{corr_flag}'
stat_img = 'test_pix.nii.gz'
display = plotting.plot_anat(t1_fname,draw_cross=False,
                  title="plot_roi", cut_coords=[25.79, 11.94, 13.47])
coords = [(25.79, 11.94, 13.47)]
display.add_markers(coords, marker_color='y', marker_size=5)
display.savefig(marker_name, dpi=500)

seed = 0
for i, val in enumerate(sources):
    if (val == np.array([104,  98, 122])).all():
        print(i)
        seed = i
        
a = np.load(f'/home/senthil/Downloads/tmp/corr_ortho_{corr_flag}.npy')
b = a[seed,:]

for idx, val in enumerate(sources):
    i, j, k = int(val[0]), int(val[1]), int(val[2])
    img[i][j][k] = b[idx]

img[img < 0.0] = 0.01 #sys.float_info.epsilon

pos_slice = []
for i, slice in enumerate(img):
    if sum(map(sum, slice)) > 0.0:
        pos_slice.append(i)

for i, pos in enumerate(pos_slice):
    if i+1 == len(pos_slice): break
    img[pos+1] = img[pos]
    img[pos+2] = img[pos_slice[i+1]]
        
img = np.swapaxes(img,0,1)
for i, pos in enumerate(pos_slice):
    if i+1 == len(pos_slice): break
    img[pos+1] = img[pos]
    img[pos+2] = img[pos_slice[i+1]]    
img = np.swapaxes(img,1,0)
   
img = np.swapaxes(img,0,2)
for i, pos in enumerate(pos_slice):
    if i+1 == len(pos_slice): break
    img[pos+1] = img[pos]
    img[pos+2] = img[pos_slice[i+1]]    
img = np.swapaxes(img,2,0)

#---------------------- Convolution Nearest Neighbor-------------------------#

final_list = [] # Axial
for dim_0_slice in img:
    output = eight_neighbor_average_convolve2d(dim_0_slice)
    final_list.append(output)
img = np.dstack(final_list)

final_list = [] # Coronal
img = np.swapaxes(img,0,1)
for dim_0_slice in img:
    output = eight_neighbor_average_convolve2d(dim_0_slice)
    final_list.append(output)
img = np.dstack(final_list)

final_list = [] # Sagittal
img = np.swapaxes(img,0,2)
for dim_0_slice in img:
    output = eight_neighbor_average_convolve2d(dim_0_slice)
    final_list.append(output)
img = np.dstack(final_list)


# Save the coorelation map to nifti image
affine = t1.affine
hdr = t1.header
result_img = nib.Nifti1Image(img, affine, header=hdr)
result_img.to_filename('test_pix.nii.gz')

# Plot the correlation on T1 image
output_html = f'/home/senthil/Downloads/tmp/corr_{corr_flag}.html'
html_view = plotting.view_img(stat_img, bg_img=t1_fname,threshold=0.0, 
                          cmap='rainbow', symmetric_cmap=False, opacity=0.6)
html_view.save_as_html(output_html)
html_view.open_in_browser()

'''
  vertex indices ( every index value for vertices will
  select a coordinate from lh_surf_coord )
 
  lh_data source data is mapped to lh_ver_idx which is then mapped to lh_surf_coord
 
  Each triangle lh_triangle_idx consist of 3 vertices
 
'''




