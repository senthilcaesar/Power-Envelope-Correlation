import mne
import numpy as np
import os
from datetime import datetime 
import os.path as op
import subprocess
from mne.transforms import apply_trans
import nibabel as nib
from pathlib import Path
import subprocess
import pathlib
import math
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from mne.connectivity import envelope_coherence
from mne.beamformer import make_lcmv, apply_lcmv_raw
import matplotlib.pyplot as plt
os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API'] = 'pyqt'
os.environ['QT_DEBUG_PLUGINS']='0'


def compute_SourceSpace(subject, subjects_dir, src_fname, source_voxel_coords, plot=True, ss='volume', 
                        volume_spacing=10):

    src = None
    if ss == 'surface':
        src = mne.setup_source_space(subject, spacing='ico5', add_dist=None,
                                subjects_dir=subjects_dir)
        src.save(src_fname, overwrite=True)
        if plot:
            mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                        src=src, orientation='coronal')
    elif ss == 'volume':
        surface = op.join(subjects_dir, subject, 'bem', 'inner_skull.surf')
        src = mne.setup_volume_source_space(subject, subjects_dir=subjects_dir,
                                        pos=volume_spacing, surface=surface, verbose=True)
        src.save(src_fname, overwrite=True)
        if plot:
            fig = mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                 brain_surfaces='white', src=src, orientation='coronal', show=True)
            plt.close()
            old_file_name = f'{subjects_dir}/{subject}/mne_files/coords.pkl'
            bashCommand = f'mv {old_file_name} {source_voxel_coords}'
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

    return src


def forward_model(subject, subjects_dir, fname_meg, trans, src, fwd_fname):
    #conductivity = (0.3, 0.006, 0.3)  # for three layers
    conductivity = (0.3,) # for single layer
    model = mne.make_bem_model(subject=subject, ico=4,
                            conductivity=conductivity,
                            subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    fwd = mne.make_forward_solution(fname_meg, trans=trans, src=src, bem=bem,
                                    meg=True, eeg=False, mindist=5.0, n_jobs=16)
    # print(fwd)
    mne.write_forward_solution(fwd_fname, fwd, overwrite=True, verbose=None)
    #leadfield = fwd['sol']['data']
    #print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)
    # np.save(f'{subjects_dir}/{subject}/mne_files/{subject}_GainMatrix.npy', leadfield)


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
    freesurfer_mri = apply_trans(vox_mri_t, VOXEL)/1e3

    return freesurfer_mri


def source_to_MNI(subject, subjects_dir, t1, sources):
     # MNI to Native scanner RAS

    ras_mni_t = mne.transforms.read_ras_mni_t(subject, subjects_dir)
    ras_mni_t = ras_mni_t['trans']
    
    # Voxel to RAS to MNI
    vox_ras_mni_t = np.dot(ras_mni_t, t1.affine)
    sources_mni = apply_trans(vox_ras_mni_t, sources)
    return sources_mni



cases = '/home/senthil/caesar/camcan/cc700/freesurfer_output/18to30.txt'
subjects_dir = '/home/senthil/caesar/camcan/cc700/freesurfer_output'
with open(cases) as f:
     case_list = f.read().splitlines()

'''Bilateral sensory locations in MNI space'''
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

log_range = np.arange(2,7.25,0.25)
carrier_freqs = [math.pow(2,val) for val in log_range]

space = 'volume'
volume_spacing = 30

start_t = datetime.now()
for freq in carrier_freqs:
    frequency = str(freq)
    for subject in case_list:
        DATA_DIR = Path(f'{subjects_dir}', f'{subject}', 'mne_files')
        bem_check = f'{subjects_dir}/{subject}/bem/'
        eye_proj1 = f'{DATA_DIR}/{subject}_eyes1-proj.fif.gz'
        eye_proj2 = f'{DATA_DIR}/{subject}_eyes2-proj.fif.gz'
        fname_meg = f'{DATA_DIR}/{subject}_ses-rest_task-rest.fif'
        t1_fname = os.path.join(subjects_dir, subject, 'mri', 'T1.mgz')
        heartbeat_proj = f'{DATA_DIR}/{subject}_heartbeat-proj.fif.gz'
        fwd_fname = f'{DATA_DIR}/{subject}_{volume_spacing}-fwd.fif.gz'
        src_fname = f'{DATA_DIR}/{subject}_{volume_spacing}-src.fif.gz'
        cov_fname = f'{DATA_DIR}/{subject}-cov_{volume_spacing}.fif.gz'
        raw_proj = f'{DATA_DIR}/{subject}_ses-rest_task-rest_proj.fif.gz'
        source_voxel_coords = f'{DATA_DIR}/{subject}_coords_{volume_spacing}.pkl'
        coherence_file_sc = f'{DATA_DIR}/{subject}_coh_{volume_spacing}_{frequency}_sc.npy'
        coherence_file_ac = f'{DATA_DIR}/{subject}_coh_{volume_spacing}_{frequency}_ac.npy'
        coherence_file_vc = f'{DATA_DIR}/{subject}_coh_{volume_spacing}_{frequency}_vc.npy'
        trans = f'/home/senthil/caesar/camcan/cc700/camcan_coreg-master/trans/{subject}-trans.fif' # The transformation file obtained by coregistration

        file_trans = pathlib.Path(trans)
        file_ss = pathlib.Path(src_fname)
        file_fm = pathlib.Path(fwd_fname)
        file_proj = pathlib.Path(raw_proj)
        file_cov = pathlib.Path(cov_fname)
        isdir_bem = pathlib.Path(bem_check)

        file_sc = pathlib.Path(coherence_file_sc)
        file_ac = pathlib.Path(coherence_file_ac)
        file_vc = pathlib.Path(coherence_file_vc)
        t1 = nib.load(t1_fname)

        if not file_trans.exists():
            print (f'{trans} File doesnt exist...')

        info = mne.io.read_info(fname_meg)

        if not file_ss.exists():
            src = compute_SourceSpace(subject, subjects_dir, src_fname, source_voxel_coords, plot=True, ss=space, 
                                volume_spacing=volume_spacing)
            seed_l_sc = MNI_to_MRI(subject, subjects_dir, t1, ROI_mni['SSC_Left'])
            seed_r_sc = MNI_to_MRI(subject, subjects_dir, t1, ROI_mni['SSC_Right'])
            seed_l_ac = MNI_to_MRI(subject, subjects_dir, t1, ROI_mni['AC_Left'])
            seed_r_ac = MNI_to_MRI(subject, subjects_dir, t1, ROI_mni['AC_Right'])
            seed_l_vc = MNI_to_MRI(subject, subjects_dir, t1, ROI_mni['VC_Left'])
            seed_r_vc = MNI_to_MRI(subject, subjects_dir, t1, ROI_mni['VC_Right'])
            some = np.where(src[0]['inuse'] == 1)
            loc_l_sc = some[0][0]
            loc_r_sc = some[0][1]
            loc_l_ac = some[0][2]
            loc_r_ac = some[0][3]
            loc_l_vc = some[0][4]
            loc_r_vc = some[0][5]
            src[0]['rr'][loc_l_sc] = seed_l_sc
            src[0]['rr'][loc_r_sc] = seed_r_sc
            src[0]['rr'][loc_l_ac] = seed_l_ac
            src[0]['rr'][loc_r_ac] = seed_r_ac
            src[0]['rr'][loc_l_vc] = seed_l_vc
            src[0]['rr'][loc_r_vc] = seed_r_vc
            src.save(src_fname, overwrite=True)
        src = mne.read_source_spaces(src_fname)

        if not file_fm.exists():
            forward_model(subject, subjects_dir, fname_meg, trans, src, fwd_fname)
        fwd = mne.read_forward_solution(fwd_fname)

        raw = mne.io.read_raw_fif(fname_meg, verbose='error', preload=True)
        n_time_samps = raw.n_times
        time_secs = raw.times
        ch_names = raw.ch_names
        n_chan = len(ch_names)
        print('\n')
        print('-------------------------- Data summary-------------------------------')
        print(f'Subject {subject}')
        print(f"The raw data object has {n_time_samps} time samples and {n_chan} channels.")
        print(f"The last time sample at {time_secs[-1]} seconds.")
        print(f"The first few channel names are {ch_names[:3]}")
        print(f"Bad channels marked during data acquisition {raw.info['bads']}")
        print(f"Sampling Frequency (No of time points/sec) {raw.info['sfreq']} Hz")
        print(f"Miscellaneous acquisition info {raw.info['description']}")
        print(f"Convert time in sec ( 60s ) to ingeter index {raw.time_as_index(60)}") # Convert time to indices
        print('------------------------------------------------------------------------')
        print('\n')
        # raw.plot(n_channels=10, scalings='auto', title='Data from arrays', show=True, block=True)
        if not file_proj.exists():
            projs_ecg, _ = compute_proj_ecg(raw, n_grad=1, n_mag=2, ch_name='ECG063')
            projs_eog1, _ = compute_proj_eog(raw, n_grad=1, n_mag=2, ch_name='EOG061')
            projs_eog2, _ = compute_proj_eog(raw, n_grad=1, n_mag=2, ch_name='EOG062')
            print(subject)
            if projs_ecg is not None:
                mne.write_proj(heartbeat_proj, projs_ecg) # Saving projectors
                raw.info['projs'] += projs_ecg
            if projs_eog1 is not None:
                mne.write_proj(eye_proj1, projs_eog1)
                raw.info['projs'] += projs_eog1
            if projs_eog2 is not None:
                mne.write_proj(eye_proj2, projs_eog2)
                raw.info['projs'] += projs_eog2
            raw.apply_proj()
            raw.save(raw_proj, proj=True, overwrite=True)
        raw_proj_applied = mne.io.read_raw_fif(raw_proj, verbose='error', preload=True)

        print(f'High-pass filtering data at 0.5 Hz')
        raw_proj_applied.filter(l_freq=0.5, h_freq=None, method='iir')

        if not file_cov.exists():
            cov = mne.compute_raw_covariance(raw_proj_applied) # compute before band-pass of interest
            mne.write_cov(cov_fname, cov)
        cov = mne.read_cov(cov_fname) 

        do_filter = True
        l_freq = freq - 2.0
        h_freq = freq + 2.0
        if do_filter:
            print(f'Band pass filter data [{l_freq}, {h_freq}]')
            raw_proj_filtered = raw_proj_applied.filter(l_freq=l_freq, h_freq=h_freq)
            data_cov = mne.compute_raw_covariance(raw_proj_filtered)
        else:
            data_cov = cov
            raw_proj_filtered = raw_proj_applied

        seed_left_sc = 0
        seed_right_sc = 1
        seed_left_ac = 2
        seed_right_ac = 3
        seed_left_vc = 4
        seed_right_vc = 5
        
        if space == 'volume':
            filters = make_lcmv(raw_proj_filtered.info, fwd, data_cov, 0.05, cov,
                            pick_ori='max-power', weight_norm='nai')
            raw_proj_filtered_comp = raw_proj_filtered.apply_hilbert(n_jobs=6)
            stcs = apply_lcmv_raw(raw_proj_filtered_comp, filters, verbose=True)

            # Coherence
            print(f'Computing coherence for {subject}....')
            if not file_sc.exists():
                coh_sc = envelope_coherence(stcs, seed_l=seed_left_sc, seed_r=seed_right_sc, fmin=l_freq, fmax=h_freq)
                np.save(coherence_file_sc, coh_sc)
                print(coherence_file_sc)
            if not file_ac.exists():
                coh_ac = envelope_coherence(stcs, seed_l=seed_left_ac, seed_r=seed_right_ac, fmin=l_freq, fmax=h_freq)
                np.save(coherence_file_ac, coh_ac)
                print(coherence_file_ac)
            if not file_vc.exists():
                coh_vc = envelope_coherence(stcs, seed_l=seed_left_vc, seed_r=seed_right_vc, fmin=l_freq, fmax=h_freq)
                np.save(coherence_file_vc, coh_vc)
                print(coherence_file_vc)
            del stcs

time_elapsed = datetime.now() - start_t
print ('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
