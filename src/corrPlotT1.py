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
        
        
spacing = 3
corr_flag = 'true'
subjects_dir = '/home/senthil/caesar/camcan/cc700/freesurfer_output'
subject = 'sub-CC110045'
src_space_fname = f'{subjects_dir}/{subject}/mne_files/{subject}-src_{spacing}.fif.gz'
src_space = mne.read_source_spaces(src_space_fname)
lh_surf_coord = src_space[0]['rr']     # Triangle Mesh coordinates
lh_triangle_idx = src_space[0]['tris'] # traingular mesh face of 3 vertices
corr_file = f'{subjects_dir}/{subject}/mne_files/{subject}_corr_ortho_{corr_flag}_{spacing}.npy'
stat_img = f'{subjects_dir}/{subject}/mne_files/test_pix.nii.gz'

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


def MNI_to_RASandVoxel(subject, subjects_dir, t1, mni_coords):
    # MNI to Native scanner RAS
    ras_mni_t = mne.transforms.read_ras_mni_t(subject, subjects_dir)
    ras_mni_t = ras_mni_t['trans']
    mni_ras_t = np.linalg.inv(ras_mni_t)
    ras_coords = apply_trans(mni_ras_t, mni_coords)
    
    # Voxel to RAS to MNI
    vox_ras_mni_t = np.dot(ras_mni_t, t1.affine)
    mni_ras_vox_t = np.linalg.inv(vox_ras_mni_t)
    vox_coords = apply_trans(mni_ras_vox_t, mni_coords)
    vox_coords = np.round(vox_coords)
    return(ras_coords, vox_coords)

def source_to_MNI(subject, subjects_dir, t1, sources):
     # MNI to Native scanner RAS
    ras_mni_t = mne.transforms.read_ras_mni_t(subject, subjects_dir)
    ras_mni_t = ras_mni_t['trans']
    
    # Voxel to RAS to MNI
    vox_ras_mni_t = np.dot(ras_mni_t, t1.affine)
    sources_mni = apply_trans(vox_ras_mni_t, sources)
    return sources_mni
    
sources_mni = source_to_MNI(subject, subjects_dir, t1, sources)
sources_mni = np.round(sources_mni)


soma_left_MNI = np.array([-42, -26, 54])
x_range = [soma_left_MNI[0]+1, soma_left_MNI[0], soma_left_MNI[0]-1]
y_range = [soma_left_MNI[1]+1, soma_left_MNI[1], soma_left_MNI[1]-1]
z_range = [soma_left_MNI[2]+1, soma_left_MNI[2], soma_left_MNI[2]-1]

soma_left_RAS, soma_left_VOX = MNI_to_RASandVoxel(subject, subjects_dir, t1, soma_left_MNI)
x, y, z = soma_left_RAS.T

m = f'{subjects_dir}/{subject}/mne_files/{subject}_coords_3.pkl'
f = open(m, 'rb')
source_mne = pickle.load(f)
img = np.zeros([256,256,256])


# Display seed
marker_name = f'{subjects_dir}/{subject}/mne_files/marker_{corr_flag}'
display = plotting.plot_anat(t1_fname, draw_cross=False,
                  title="plot_roi", cut_coords=[x, y, z])
coords = [(x, y, z)]
display.add_markers(coords, marker_color='y', marker_size=5)
display.savefig(marker_name, dpi=500)


def create_label():
    label_name = '/home/senthil/Downloads/tmp/sub-CC221373/nilearn_files/label.mgz'
    label_mri = np.zeros([256,256,256])
    label_mri[104][98][122] = 4
    affine = t1.affine
    hdr = t1.header
    result_img = nib.freesurfer.mghformat.MGHImage(label_mri, affine, header=hdr)
    result_img.to_filename(label_name)

    
seed = 0
for i, val in enumerate(sources_mni):
    if val[0] in x_range and val[1] in y_range and val[2] in z_range:
        print(i, val)
        seed = i
        break

a = np.load(corr_file)
b = a.copy()
for idx, val in enumerate(sources):
    i, j, k = int(val[0]), int(val[1]), int(val[2])
    img[i][j][k] = b[idx]

img[img < 0.0] = 0.01 #sys.float_info.epsilon

fill_slices = True
if fill_slices:
    
    #----------------- AXIAL----------------------------------#
    pos_slice = []
    for i, slice in enumerate(img):
        if sum(map(sum, slice)) > 0.0:
            pos_slice.append(i)
    for i, pos in enumerate(pos_slice):
        if i+1 == len(pos_slice): break
        img[pos+1] = img[pos]
        img[pos+2] = img[pos_slice[i+1]]
          
    #------------------CORONAL-------------------------------#
    img = np.swapaxes(img,0,1)
    pos_slice = []
    for i, slice in enumerate(img):
        if sum(map(sum, slice)) > 0.0:
            pos_slice.append(i)
    for i, pos in enumerate(pos_slice):
        if i+1 == len(pos_slice): break
        img[pos+1] = img[pos]
        img[pos+2] = img[pos_slice[i+1]]    
    img = np.swapaxes(img,1,0)
        
    #-------------------SAGITTAL-------------------------------#
    img = np.swapaxes(img,0,2)
    pos_slice = []
    for i, slice in enumerate(img):
        if sum(map(sum, slice)) > 0.0:
            pos_slice.append(i)
    for i, pos in enumerate(pos_slice):
        if i+1 == len(pos_slice): break
        img[pos+1] = img[pos]
        img[pos+2] = img[pos_slice[i+1]]    
    img = np.swapaxes(img,2,0)

#---------------------- Convolution Nearest Neighbor-------------------------#

convolution = True
if convolution:
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


#Save the coorelation map to nifti image
html = True
if html:
    affine = t1.affine
    hdr = t1.header
    result_img = nib.Nifti1Image(img, affine, header=hdr)
    result_img.to_filename(stat_img)
    
    # Plot the correlation on T1 image
    output_html = f'{subjects_dir}/{subject}/mne_files/corr_{corr_flag}_{spacing}.html'
    print(output_html)
    html_view = plotting.view_img(stat_img, bg_img=t1_fname, #threshold=None, 
                              cmap='rainbow', symmetric_cmap=False, 
                              opacity=0.6, vmax=b.max())
    html_view.save_as_html(output_html)
    html_view.open_in_browser()



















'''
  vertex indices ( every index value for vertices will
  select a coordinate from lh_surf_coord )
 
  lh_data source data is mapped to lh_ver_idx which is then mapped to lh_surf_coord
 
  Each triangle lh_triangle_idx consist of 3 vertices
  
  src_space[0]['rr'] = The vertices of the source space
  
 
'''