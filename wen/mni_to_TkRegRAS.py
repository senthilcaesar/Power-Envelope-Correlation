import mne
import os
import numpy as np
import nibabel as nib
from mne.transforms import apply_trans

# Bilateral sensory locations in MNI space
ROI_mni = {
    'AC_Left':[-54, -22, 10],   # Auditory cortex left
    'AC_Right':[52, -24, 12],   # Auditory cortex right
    'SSC_Left':[-42, -26, 54],  # Somatosensory cortex left
    'SSC_Right':[38, -32, 48],  # Somatosensory cortex right
    'VC_Left':[-20, -86, 18],   # Visual cortex left
    'VC_Right':[16, -80, 26],   # Visual cortex right
    'MT+_Left':[-47, -69, -3],
    'MT+_Right':[54, -63, -8],
    'MTL_Left':[-20, -40, -10],
    'MTL_Right':[40, -40, 0],
    'SMC_Left':[-40, -40, -60],
    'SMC_Right':[40, -30, 50],
    'LPC_Left':[-39, -54, 32],
    'LPC_Right':[46, -45, 39],
    'DPFC_Left':[-40, 30, 50],
    'DPFC_Right':[30, 20, 30],
    'TMPC_Left':[-50, -40, -10],
    'TMPC_Right':[60, -20, 0],
    'MPFC_MidBrain':[-3, 39, -2],
    'SMA_MidBrain':[-2, 1, 51],
    }

def MNI_to_MRI(subject, subjects_dir, t1, mni_coords):
    # MNI to Native scanner RAS
    ras_mni_t = mne.transforms.read_ras_mni_t(subject, subjects_dir)
    ras_mni_t = ras_mni_t['trans']
    mni_ras_t = np.linalg.inv(ras_mni_t)
    ras_coords = apply_trans(mni_ras_t, mni_coords)

    # Voxel to RAS to MNI
    vox_ras_mni_t = np.dot(ras_mni_t, t1.affine)
    mni_ras_vox_t = np.linalg.inv(vox_ras_mni_t)

    VOXEL = apply_trans(mni_ras_vox_t, mni_coords)
    vox_mri_t = t1.header.get_vox2ras_tkr()
    freesurfer_mri = apply_trans(vox_mri_t, VOXEL)

    return ras_coords, VOXEL, freesurfer_mri

subject='sub-CC321000'
subjects_dir = '/home/senthilp/caesar/camcan/cc700/freesurfer_output'
t1_fname = os.path.join(subjects_dir, subject, 'mri', 'T1.mgz')
t1 = nib.load(t1_fname)
mni_coord = ROI_mni['VC_Left']
ras, voxel, freeSurfer_MRI = MNI_to_MRI(subject, subjects_dir, t1, mni_coord)

location = " "
print(f"{'mni coords' + location:<35} {mni_coord}")
print(f"{'voxel coords' + location:<35} {voxel}")
print(f"{'freesurfer RAS coords' + location:35} {ras}")
print(f"{'freesurfer TkReg RAS coords' + location:<35} {freeSurfer_MRI}")
