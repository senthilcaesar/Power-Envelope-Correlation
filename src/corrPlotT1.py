import pickle
import mne
import os
import nibabel
import numpy as np
from itertools import product
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
        

def create_label():
    label_name = '/home/senthil/Downloads/tmp/sub-CC221373/nilearn_files/label.mgz'
    label_mri = np.zeros([256,256,256])
    label_mri[104][98][122] = 4
    affine = t1.affine
    hdr = t1.header
    result_img = nib.freesurfer.mghformat.MGHImage(label_mri, affine, header=hdr)
    result_img.to_filename(label_name)
 

def display_seed():
    #Display seed
    marker_name = f'{subjects_dir}/{subject}/mne_files/marker_{corr_flag}'
    display = plotting.plot_anat(t1_fname, draw_cross=False,
                      title="plot_roi", cut_coords=[x, y, z])
    coords = [(x, y, z)]
    display.add_markers(coords, marker_color='y', marker_size=5)
    display.savefig(marker_name, dpi=500)

    
def source_to_MNI(subject, subjects_dir, t1, sources):
     # MNI to Native scanner RAS
     
    #subject='fsaverage'
    #subjects_dir='/home/senthil/mne_data/MNE-sample-data/subjects'
    ras_mni_t = mne.transforms.read_ras_mni_t(subject, subjects_dir)
    ras_mni_t = ras_mni_t['trans']
    
    # Voxel to RAS to MNI
    vox_ras_mni_t = np.dot(ras_mni_t, t1.affine)
    sources_mni = apply_trans(vox_ras_mni_t, sources)
    return sources_mni


def MNI_to_RASandVoxel(subject, subjects_dir, t1, mni_coords):
    # MNI to Native scanner RAS
    
    #subject='fsaverage'
    #subjects_dir='/home/senthil/mne_data/MNE-sample-data/subjects'
    
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


def neighbors(index):
    N = len(index)
    for relative_index in product((-1, 0, 1), repeat=N):
        if not all(i == 0 for i in relative_index):
            yield tuple(i + i_rel for i, i_rel in zip(index, relative_index))
            
            
def miscelaneous():
    src_space_fname = '2932.fif.gz'
    src_space_fname = '/home/senthil/mne_data/MNE-sample-data/subjects/fsaverage/bem/fsaverage-vol-5-src.fif'
    t1_fname = '/home/senthil/mne_data/MNE-sample-data/subjects/fsaverage/mri/brain.mgz'


cases = '/home/senthil/caesar/camcan/cc700/freesurfer_output/50.txt'
subjects_dir = '/home/senthil/caesar/camcan/cc700/freesurfer_output'
with open(cases) as f:
     case_list = f.read().splitlines()

spacing = 3
corr_flag = 'true'
subjects_dir = '/home/senthil/caesar/camcan/cc700/freesurfer_output'
    

for idx, case in enumerate(case_list):
    subject = case
    src_space_fname = f'{subjects_dir}/{subject}/mne_files/{subject}-src_{spacing}.fif.gz'
    src_space = mne.read_source_spaces(src_space_fname)
    lh_surf_coord = src_space[0]['rr']     # Triangle Mesh coordinates
    lh_triangle_idx = src_space[0]['tris'] # traingular mesh face of 3 vertices
    corr_file = f'{subjects_dir}/{subject}/mne_files/{subject}_corr_ortho_{corr_flag}_{spacing}_no_morph.npy'
    stat_img = f'{subjects_dir}/{subject}/mne_files/{subject}_corr_vol_{corr_flag}_{spacing}_no_morph.nii.gz'
    
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
    
    
    sources_mni = source_to_MNI(subject, subjects_dir, t1, sources)
    sources_mni = np.round(sources_mni)


    soma_left_MNI = np.array([-42, -26, 54])
    
    x1 = list(np.linspace(soma_left_MNI[0], soma_left_MNI[0]-3, 4))
    x2 = list(np.linspace(soma_left_MNI[0], soma_left_MNI[0]+2, 3))
    y1 = list(np.linspace(soma_left_MNI[1], soma_left_MNI[1]-3, 4))
    y2 = list(np.linspace(soma_left_MNI[1], soma_left_MNI[1]+2, 3))
    z1 = list(np.linspace(soma_left_MNI[2], soma_left_MNI[2]-3, 4))
    z2 = list(np.linspace(soma_left_MNI[2], soma_left_MNI[2]+2, 3))
    
    x_range = set(x1+x2)
    y_range = set(y1+y2)
    z_range = set(z1+z2)

    soma_left_RAS, soma_left_VOX = MNI_to_RASandVoxel(subject, subjects_dir, t1, soma_left_MNI)
    x, y, z = soma_left_RAS.T
    print(x,y,z)
    
    m = f'{subjects_dir}/{subject}/mne_files/{subject}_coords_{spacing}.pkl'
    f = open(m, 'rb')
    source_mne = pickle.load(f)
    img = np.zeros([256,256,256])

    seed = 0
    for i, val in enumerate(sources_mni):
        if val[0] in x_range and val[1] in y_range and val[2] in z_range:
            print(i, val)
            seed = i
    
    a = np.load(corr_file)
    for idx, val in enumerate(sources):
        i, j, k = int(val[0]), int(val[1]), int(val[2])
        img[i][j][k] = a[idx]
    img[img < 0.0] = 0.01 #sys.float_info.epsilon


    # Fill neiboring voxels
    found = np.argwhere(img > 0.0)
    for a1,b1,c1 in found:
        neighbor = list(neighbors((a1, b1, c1)))
        for i,j,k in neighbor:
            img[i][j][k] = img[a1][b1][c1]
                                 

    fill_slices = True
    if fill_slices:
        
        #----------------- AXIAL----------------------------------#
        pos_slice_axial = []
        for i, slice in enumerate(img):
            if sum(map(sum, slice)) > 0.0:
                pos_slice_axial.append(i)
        for i, pos in enumerate(pos_slice_axial):
            if i+1 == len(pos_slice_axial):break
            val = pos_slice_axial[i+1] - pos
            if val == 6:
                img[pos+1] = img[pos]
                img[pos+2] = img[pos] 
                img[pos+3] = img[pos]
                img[pos+4] = img[pos_slice_axial[i+1]]
                img[pos+5] = img[pos_slice_axial[i+1]]
                img[pos+6] = img[pos_slice_axial[i+1]]
                
            if val == 7:
                img[pos+1] = img[pos]
                img[pos+2] = img[pos] 
                img[pos+3] = img[pos]
                img[pos+4] = img[pos_slice_axial[i+1]]
                img[pos+5] = img[pos_slice_axial[i+1]]
                img[pos+6] = img[pos_slice_axial[i+1]]
                img[pos+7] = img[pos_slice_axial[i+1]]
            
            if val == 3:
                img[pos+1] = img[pos]
                img[pos+2] = img[pos_slice_axial[i+1]]
                
              
        #------------------CORONAL-------------------------------#
        img = np.swapaxes(img,0,1)
        pos_slice_coronal = []
        for i, slice in enumerate(img):
            if sum(map(sum, slice)) > 0.0:
                pos_slice_coronal.append(i)
        for i, pos in enumerate(pos_slice_coronal):
            if i+1 == len(pos_slice_coronal): break
            val = pos_slice_coronal[i+1] - pos
            if val == 6:
                img[pos+1] = img[pos]
                img[pos+2] = img[pos] 
                img[pos+3] = img[pos]
                img[pos+4] = img[pos_slice_coronal[i+1]]
                img[pos+5] = img[pos_slice_coronal[i+1]]
                img[pos+6] = img[pos_slice_coronal[i+1]]
                
            if val == 7:
                img[pos+1] = img[pos]
                img[pos+2] = img[pos] 
                img[pos+3] = img[pos]
                img[pos+4] = img[pos_slice_coronal[i+1]]
                img[pos+5] = img[pos_slice_coronal[i+1]]
                img[pos+6] = img[pos_slice_coronal[i+1]]
                img[pos+7] = img[pos_slice_coronal[i+1]]
            
            if val == 3:
                img[pos+1] = img[pos]
                img[pos+2] = img[pos_slice_coronal[i+1]]
    
            
        img = np.swapaxes(img,1,0)
            
        #-------------------SAGITTAL-------------------------------#
        img = np.swapaxes(img,0,2)
        pos_slice_sagittal = []
        for i, slice in enumerate(img):
            if sum(map(sum, slice)) > 0.0:
                pos_slice_sagittal.append(i)
        for i, pos in enumerate(pos_slice_sagittal):
            if i+1 == len(pos_slice_sagittal): break
        
            val = pos_slice_sagittal[i+1] - pos
            if val == 6:
                img[pos+1] = img[pos]
                img[pos+2] = img[pos] 
                img[pos+3] = img[pos]
                img[pos+4] = img[pos_slice_sagittal[i+1]]
                img[pos+5] = img[pos_slice_sagittal[i+1]]
                img[pos+6] = img[pos_slice_sagittal[i+1]]
                
            if val == 7:
                img[pos+1] = img[pos]
                img[pos+2] = img[pos] 
                img[pos+3] = img[pos]
                img[pos+4] = img[pos_slice_sagittal[i+1]]
                img[pos+5] = img[pos_slice_sagittal[i+1]]
                img[pos+6] = img[pos_slice_sagittal[i+1]]
                img[pos+7] = img[pos_slice_sagittal[i+1]]
     
            if val == 3:
                img[pos+1] = img[pos]
                img[pos+2] = img[pos_slice_sagittal[i+1]]
                
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
    
    
    affine = t1.affine
    hdr = t1.header
    result_img = nib.Nifti1Image(img, affine, header=hdr)
    result_img.to_filename(stat_img)
    
    
    #Save the coorelation map to nifti image
    html = False
    if html:
        # Plot the correlation on T1 image
        output_html = f'{subjects_dir}/{subject}/mne_files/corr_{corr_flag}_{spacing}_without_morph.html'
        print(output_html)
        html_view = plotting.view_img(stat_img, bg_img=t1_fname, #threshold=None, 
                                  cmap='rainbow', symmetric_cmap=False, 
                                  opacity=0.6, vmax=a.max(), cut_coords=(x,y,z))
        html_view.save_as_html(output_html)
        html_view.open_in_browser()

    some = nib.load(stat_img)
    some.set_data_dtype('uint8')
    some_data = some.get_fdata()
    print(some_data.max(), img.max())
    print(f'Case {idx} completed...')
    print('/n')

















'''
  vertex indices ( every index value for vertices will
  select a coordinate from lh_surf_coord )
 
  lh_data source data is mapped to lh_ver_idx which is then mapped to lh_surf_coord
 
  Each triangle lh_triangle_idx consist of 3 vertices
  
  src_space[0]['rr'] = The vertices of the source space
  
 
'''
