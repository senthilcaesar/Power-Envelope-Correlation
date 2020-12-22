from re import VERBOSE
import mne
import numpy as np
import os
import sys
import os.path as op
import subprocess
from mne.transforms import apply_trans
import nibabel as nib
import multiprocessing as mp
from pathlib import Path
import subprocess
import pathlib
from mne.connectivity import envelope_correlation, envelope_coherence
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from mne.beamformer import make_lcmv, apply_lcmv_raw, apply_lcmv_epochs
from functools import wraps
import matplotlib.pyplot as plt
import time
os.environ['ETS_TOOLKIT']='qt4'
os.environ['QT_API']='pyqt'
os.environ['QT_DEBUG_PLUGINS']='0'



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


freqs = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]

def convert(seconds): 
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print (f'@timefn: {fn.__name__} took {convert(t2-t1)} (hh:mm:ss)')
        return result
    return measure_time


def compute_SourceSpace(subject, subjects_dir, src_fname, source_voxel_coords, plot=False, ss='volume', 
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
                                    meg=True, eeg=False, mindist=5.06)
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


def run_correlation(subjects_dir, subject, volume_spacing, freq):

    os.environ["OMP_NUM_THREADS"] = "10"         # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"] = "10"    # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"] = "10"         # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = "10"  # export VECLIB_MAXIMUM_THREADS=1
    os.environ["NUMEXPR_NUM_THREADS"] = "10"     # export NUMEXPR_NUM_THREADS=1

    frequency = str(freq)
    DATA_DIR = Path(f'{subjects_dir}', f'{subject}', 'mne_files')
    eye_proj1 = f'{DATA_DIR}/{subject}_eyes1-proj.fif.gz'
    eye_proj2 = f'{DATA_DIR}/{subject}_eyes2-proj.fif.gz'
    fname_meg = f'{DATA_DIR}/{subject}_ses-rest_task-rest.fif'
    t1_fname = os.path.join(subjects_dir, subject, 'mri', 'T1.mgz')
    heartbeat_proj = f'{DATA_DIR}/{subject}_heartbeat-proj.fif.gz'
    fwd_fname = f'{DATA_DIR}/{subject}_{volume_spacing}-fwd.fif.gz'
    src_fname = f'{DATA_DIR}/{subject}_{volume_spacing}-src.fif.gz'
    cov_fname = f'{DATA_DIR}/{subject}-cov_{volume_spacing}.fif.gz'
    raw_cov_fname = f'{DATA_DIR}/{subject}-rawcov_{volume_spacing}.fif.gz'
    raw_proj = f'{DATA_DIR}/{subject}_ses-rest_task-rest_proj.fif.gz'
    source_voxel_coords = f'{DATA_DIR}/{subject}_coords_{volume_spacing}.pkl'
    corr_true_file_acLeft_wholebrain = f'{DATA_DIR}/{subject}_corr_ortho_true_{volume_spacing}_{frequency}_acLeft_wholebrain.npy'
    corr_true_file_scLeft_wholebrain = f'{DATA_DIR}/{subject}_corr_ortho_true_{volume_spacing}_{frequency}_scLeft_wholebrain.npy'
    corr_true_file_vcLeft_wholebrain = f'{DATA_DIR}/{subject}_corr_ortho_true_{volume_spacing}_{frequency}_vcLeft_wholebrain.npy'
    corr_true_file_acRight_wholebrain = f'{DATA_DIR}/{subject}_corr_ortho_true_{volume_spacing}_{frequency}_acRight_wholebrain.npy'
    corr_true_file_scRight_wholebrain = f'{DATA_DIR}/{subject}_corr_ortho_true_{volume_spacing}_{frequency}_scRight_wholebrain.npy'
    corr_true_file_vcRight_wholebrain = f'{DATA_DIR}/{subject}_corr_ortho_true_{volume_spacing}_{frequency}_vcRight_wholebrain.npy'

    check_for_files = []
    check_for_files.append(corr_true_file_acLeft_wholebrain)
    check_for_files.append(corr_true_file_scLeft_wholebrain)
    check_for_files.append(corr_true_file_vcLeft_wholebrain)
    check_for_files.append(corr_true_file_acRight_wholebrain)
    check_for_files.append(corr_true_file_scRight_wholebrain)
    check_for_files.append(corr_true_file_vcRight_wholebrain)

    file_exist = [f for f in check_for_files if os.path.isfile(f)]
    file_not_exist = list(set(file_exist) ^ set(check_for_files))

    if not file_not_exist:
        print('SC, AC, VC correlation files exists...')

    else:
        trans = f'/home/senthilp/caesar/camcan/cc700/camcan_coreg-master/trans/{subject}-trans.fif' # The transformation file obtained by coregistration
        file_trans = pathlib.Path(trans)
        file_ss = pathlib.Path(src_fname)
        file_fm = pathlib.Path(fwd_fname)
        file_proj = pathlib.Path(raw_proj)
        file_cov = pathlib.Path(cov_fname)
        file_rawcov = pathlib.Path(raw_cov_fname)
        t1 = nib.load(t1_fname)

        if not file_trans.exists():
            print (f'{trans} File doesnt exist...')
            sys.exit(0)

        info = mne.io.read_info(fname_meg)
        # plot_registration(info, trans, subject, subjects_dir)

        if not file_ss.exists():
            src = compute_SourceSpace(subject, subjects_dir, src_fname, source_voxel_coords, plot=True, ss='volume', 
                                volume_spacing=volume_spacing)
            seed_l_sc = MNI_to_MRI(subject, subjects_dir, t1, ROI_mni['SSC_Left'])
            seed_r_sc = MNI_to_MRI(subject, subjects_dir, t1, ROI_mni['SSC_Right'])
            seed_l_ac = MNI_to_MRI(subject, subjects_dir, t1, ROI_mni['AC_Left'])
            seed_r_ac = MNI_to_MRI(subject, subjects_dir, t1, ROI_mni['AC_Right'])
            seed_l_vc = MNI_to_MRI(subject, subjects_dir, t1, ROI_mni['VC_Left'])
            seed_r_vc = MNI_to_MRI(subject, subjects_dir, t1, ROI_mni['VC_Right'])
            src_inuse = np.where(src[0]['inuse'] == 1)
            loc_l_sc = src_inuse[0][0]
            loc_r_sc = src_inuse[0][1]
            loc_l_ac = src_inuse[0][2]
            loc_r_ac = src_inuse[0][3]
            loc_l_vc = src_inuse[0][4]
            loc_r_vc = src_inuse[0][5]
            src[0]['rr'][loc_l_sc] = seed_l_sc
            src[0]['rr'][loc_r_sc] = seed_r_sc
            src[0]['rr'][loc_l_ac] = seed_l_ac
            src[0]['rr'][loc_r_ac] = seed_r_ac
            src[0]['rr'][loc_l_vc] = seed_l_vc
            src[0]['rr'][loc_r_vc] = seed_r_vc
            src.save(src_fname, overwrite=True)
        src = mne.read_source_spaces(src_fname)
        #view_SS_brain(subject, subjects_dir, src)

        if not file_fm.exists():
            forward_model(subject, subjects_dir, fname_meg, trans, src, fwd_fname)
        fwd = mne.read_forward_solution(fwd_fname)

        # sensitivty_plot(subject, subjects_dir, fwd)
        raw = mne.io.read_raw_fif(fname_meg, verbose='error', preload=True)

        srate = raw.info['sfreq']
        n_time_samps = raw.n_times
        time_secs = raw.times
        ch_names = raw.ch_names
        n_chan = len(ch_names)
        freq_res =  srate/n_time_samps
        print('\n')
        print('-------------------------- Data summary-------------------------------')
        print(f'Subject {subject}')
        print(f"Frequency resolution {freq_res} Hz")
        print(f"The first few channel names are {ch_names[:3]}")
        print(f"The last time sample at {time_secs[-1]} seconds.")
        print(f"Sampling Frequency (No of time points/sec) {srate} Hz")
        print(f"Miscellaneous acquisition info {raw.info['description']}")
        print(f"Bad channels marked during data acquisition {raw.info['bads']}")
        print(f"Convert time in sec ( 60s ) to ingeter index {raw.time_as_index(60)}") # Convert time to indices
        print(f"The raw data object has {n_time_samps} time samples and {n_chan} channels.")
        print('------------------------------------------------------------------------')
        print('\n')
        # raw.plot(n_channels=10, scalings='auto', title='Data from arrays', show=True, block=True)
        if not file_proj.exists():
            projs_ecg, _ = compute_proj_ecg(raw, n_grad=1, n_mag=2, ch_name='ECG063')
            projs_eog1, _ = compute_proj_eog(raw, n_grad=1, n_mag=2, ch_name='EOG061')
            projs_eog2, _ = compute_proj_eog(raw, n_grad=1, n_mag=2, ch_name='EOG062')
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

        # cov.plot(raw.info, proj=True, exclude='bads', show_svd=False
        # raw_proj_applied.crop(tmax=10)
        
        do_epochs = False

        l_freq = freq-2.0
        h_freq = freq+2.0
        print(f'Band pass filter data [{l_freq}, {h_freq}]')
        raw_proj_filtered = raw_proj_applied.filter(l_freq=l_freq, h_freq=h_freq)

        if do_epochs:
            print('Segmenting raw data...')
            events = mne.make_fixed_length_events(raw_proj_filtered, duration=5.)
            raw_proj_filtered = mne.Epochs(raw_proj_filtered, events=events, tmin=0, tmax=5.,
                                            baseline=None, preload=True)
            data_cov = mne.compute_covariance(raw_proj_filtered)         
        else:
            if not file_rawcov.exists():
                data_cov = mne.compute_raw_covariance(raw_proj_filtered)
                mne.write_cov(raw_cov_fname, data_cov)
            else:
                data_cov = mne.read_cov(file_rawcov)


        seed_left_sc = 0
        seed_right_sc = 1
        seed_left_ac = 2
        seed_right_ac = 3
        seed_left_vc = 4
        seed_right_vc = 5
        

        filters = make_lcmv(raw_proj_filtered.info, fwd, data_cov, 0.05, cov,
                            pick_ori='max-power', weight_norm='nai')
        raw_proj_filtered_comp = raw_proj_filtered.apply_hilbert()

        if do_epochs:
            stcs = apply_lcmv_epochs(raw_proj_filtered_comp, filters, return_generator=False)
        else:
            stcs = apply_lcmv_raw(raw_proj_filtered_comp, filters, verbose=True)
            stcs = [stcs]
        # Power Envelope Correlation
        print(f'Computing Power Envelope Correlation for {subject}....Orthogonalize True')

        all_corr = envelope_correlation(stcs, combine=None, orthogonalize="pairwise",
                    log=True, absolute=True, verbose=None)

        np.save(corr_true_file_scLeft_wholebrain, all_corr[seed_left_sc])
        np.save(corr_true_file_acLeft_wholebrain, all_corr[seed_left_ac])
        np.save(corr_true_file_vcLeft_wholebrain, all_corr[seed_left_vc])

        np.save(corr_true_file_scRight_wholebrain, all_corr[seed_right_sc])
        np.save(corr_true_file_acRight_wholebrain, all_corr[seed_right_ac])
        np.save(corr_true_file_vcRight_wholebrain, all_corr[seed_right_vc])

        del stcs


cases = '/home/senthilp/caesar/camcan/cc700/freesurfer_output/68to88.txt'
subjects_dir = '/home/senthilp/caesar/camcan/cc700/freesurfer_output'
with open(cases) as f:
     case_list = f.read().splitlines()

@timefn
def main():
    volume_spacing = 7.8
    for freq in freqs:
        print(f'Data filtered at frequency {str(freq)} Hz...')
        pool = mp.Pool(processes=7)
        for subject in case_list:
            pool.apply_async(run_correlation, args=[subjects_dir, subject, volume_spacing, freq])
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
