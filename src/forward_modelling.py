from operator import ne
from pickle import NONE
from re import VERBOSE
import mne
from mne.io import proj
from mne.utils.docs import stc
import numpy as np
import os
import sys
from datetime import datetime 
import os.path as op
import subprocess
from mne.transforms import apply_trans
import nibabel as nib
import multiprocessing as mp
from multiprocessing import Manager
from pathlib import Path
from numpy.core.shape_base import block
from surfer import Brain
from IPython.display import Image
from mayavi import mlab
import subprocess
import pickle
import pathlib
from mne.time_frequency import tfr_morlet
from mne.viz import plot_alignment, set_3d_view
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from mne.connectivity import envelope_correlation
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
import matplotlib.pyplot as plt
from surfer.utils import verbose
os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API'] = 'pyqt'


def compute_bem(subject, subjects_dir):
    mne.bem.make_watershed_bem(subject=subject, atlas=True, brainmask='ws.mgz',
                              subjects_dir=subjects_dir, overwrite=True)


def plot_bem(subject, subjects_dir):
    mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir, 
                 orientation='coronal')


def compute_scalp_surfaces(subject, subjects_dir):
    bashCommand = f'mne make_scalp_surfaces --overwrite -d {subjects_dir} -s {subject} --no-decimate --force'
    subprocess.check_output(bashCommand, shell=True)


def coregistration(subject, subjects_dir, trans):
    mne.gui.coregistration(subject=subject, subjects_dir=subjects_dir)


def plot_registration(info, trans, subject, subjects_dir):
    fig = plot_alignment(info, trans, subject=subject, dig=True,
                        meg=True, subjects_dir=subjects_dir,
                        coord_frame='head')
    set_3d_view(figure=fig, azimuth=135, elevation=80)
    mlab.savefig('/home/senthil/Desktop/coreg.jpg')
    Image(filename='/home/senthil/Desktop/coreg.jpg', width=500)
    mlab.show()


def view_SS_brain(subject, subjects_dir, src):
    brain = Brain(subject, 'lh', 'white', subjects_dir=subjects_dir)
    surf = brain.geo['lh']
    vertidx = np.where(src[0]['inuse'])[0]
    mlab.points3d(surf.x[vertidx], surf.y[vertidx],
                surf.z[vertidx], color=(1, 1, 0), scale_factor=1.5)
    mlab.savefig('source_space_subsampling.jpg')
    Image(filename='source_space_subsampling.jpg', width=500)
    mlab.show()

    brain = Brain(subject, 'lh', 'inflated', subjects_dir=subjects_dir)
    surf = brain.geo['lh']
    mlab.points3d(surf.x[vertidx], surf.y[vertidx],
                surf.z[vertidx], color=(1, 1, 0), scale_factor=1.5)
    mlab.savefig('source_space_subsampling2.jpg')
    Image(filename='source_space_subsampling2.jpg', width=500)
    mlab.show()


def compute_SourceSpace(subject, subjects_dir, src_fname, source_voxel_coords, plot=True, ss='surface', 
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
    leadfield = fwd['sol']['data']
    print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)
    # np.save(f'{subjects_dir}/{subject}/mne_files/{subject}_GainMatrix.npy', leadfield)


def sensitivty_plot(subject, subjects_dir, fwd):
    leadfield = fwd['sol']['data']
    grad_map = mne.sensitivity_map(fwd, ch_type='grad', mode='fixed')
    mag_map = mne.sensitivity_map(fwd, ch_type='mag', mode='fixed')
    picks_meg = mne.pick_types(fwd['info'], meg=True, eeg=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Lead field matrix (500 dipoles only)', fontsize=18)
    for ax, picks, ch_type in zip(axes, [picks_meg], ['meg']):
        im = ax.imshow(leadfield[picks, :500], origin='lower', aspect='auto',
                    cmap='RdBu_r')
        ax.set_title(ch_type.upper(), fontsize=18)
        ax.set_xlabel('sources',fontsize = 18)
        ax.set_ylabel('sensors',fontsize = 18)
        fig.colorbar(im, ax=ax, cmap='RdBu_r')

    fig_2, ax = plt.subplots()
    ax.hist([grad_map.data.ravel(), mag_map.data.ravel()],
            bins=20, label=['Gradiometers', 'Magnetometers'],
            color=['c', 'b'])
    fig_2.legend()
    ax.set_title('Normal orientation sensitivity', fontsize=18)
    ax.set_xlabel('sensitivity',fontsize = 18)
    ax.set_ylabel('count',fontsize = 18)
    plt.show()

    # Inflated sensivity map
    clim = dict(kind='percent', lims=(0.0, 50, 95), smoothing_steps=3)  # let's see single dipoles
    brain = grad_map.plot(subject=subject, time_label='GRAD sensitivity', surface='inflated',
                        subjects_dir=subjects_dir, clim=clim, smoothing_steps=8, alpha=0.85)
    mlab.show()
    view = 'lat'
    brain.show_view(view)
    brain.save_image(f'sensitivity_map_grad_{view}.jpg')
    Image(filename=f'sensitivity_map_grad_{view}.jpg', width=400)


def view_source_origin(corr, labels, inv, subjects_dir):
    threshold_prop = 0.15
    degree = mne.connectivity.degree(corr, threshold_prop=threshold_prop)
    stc = mne.labels_to_stc(labels, degree)
    stc = stc.in_label(mne.Label(inv['src'][0]['vertno'], hemi='lh') +
                    mne.Label(inv['src'][1]['vertno'], hemi='rh'))
    brain = stc.plot(colormap='gnuplot',
        subjects_dir=subjects_dir, views='dorsal', hemi='both',
        smoothing_steps=25, time_label='Beta band')
    mlab.show()


def save_source_estimates(stcs, subjects_dir, subject, volume_spacing):

    for idx, se in enumerate(stcs):
        stcs_fname = f'{subjects_dir}/{subject}/mne_files/{subject}_{volume_spacing}-stcs_epoch{idx}'
        print(f'Saving epoch {idx} source estimate to file {stcs_fname}....')
        se.save(stcs_fname, ftype='h5')


def plot_psd(epochs):
    # Plot power spectral density
    # Exploring frequency content of our epochs
    epochs.plot_psd(average=True, spatial_colors=False, fmin=0, fmax=50)
    epochs.plot_psd_topomap(normalize=True)


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


def locate_seed(interest, sources_mni):
        x1 = list(np.linspace(interest[0], interest[0]-3, 4))
        x2 = list(np.linspace(interest[0], interest[0]+2, 3))
        y1 = list(np.linspace(interest[1], interest[1]-3, 4))
        y2 = list(np.linspace(interest[1], interest[1]+2, 3))
        z1 = list(np.linspace(interest[2], interest[2]-3, 4))
        z2 = list(np.linspace(interest[2], interest[2]+3, 4))

        x_range = set(x1+x2)
        y_range = set(y1+y2)
        z_range = set(z1+z2)

        cord = 0
        seed = 0
        seeds_nearby = []
        min_dist = 100
        for i, val in enumerate(sources_mni):
            if val[0] in x_range and val[1] in y_range and val[2] in z_range:
                seed, cord = i, val
                #print(i, val)
                seeds_nearby.append((i, val))
        if len(seeds_nearby) == 1:
            print(f'Seed MNI location {seeds_nearby[0]}')
            return seeds_nearby[0][0]
        else:
            '''
            Find the nearest neighbor
            '''
            for neighbor in seeds_nearby:
                p1 = np.array([interest[0], interest[1], interest[2]])
                p2 = np.array([neighbor[1][0], neighbor[1][1],  neighbor[1][2]])
                squared_dist = np.sum((p1-p2)**2, axis=0)
                dist = np.sqrt(squared_dist)
                if dist < min_dist: min_dist = dist; seed = neighbor[0]
            print(f'Seed MNI location {seed}... Min distance {min_dist}')
            return seed



cases = '/home/senthil/caesar/camcan/cc700/freesurfer_output/10.txt'
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

freqs = {
        2: [0, 4],
        4: [2, 6],
        6: [4, 8],
        8: [6, 10],
        10: [8, 12],
        12: [10, 14],
        14: [12, 16],
        16: [14, 18],
        18: [16, 20],
        20: [18, 22],
        24: [22, 26],
        28: [26, 30],
        32: [30, 34],
        42: [40, 44],
        52: [50, 54],
        64: [62, 66],
        96: [94, 98],
        128: [126, 130]
}
#fname_src_fsaverage = '/home/senthil/mne_data/MNE-sample-data/subjects/fsaverage/bem/fsaverage-vol-5-src.fif'
#fname_src_fsaverage = '/home/senthil/Downloads/2932.fif.gz'
#fname_t1_fsaverage = '/home/senthil/mne_data/MNE-sample-data/subjects/fsaverage/mri/brain.mgz'

start_t = datetime.now()

for key in freqs:
    frequency = str(key)
    print(f'Data filtered at frequency {frequency} Hz...')
    for case in case_list:
        subject = case
        space = 'volume'
        volume_spacing = 5
        DATA_DIR = Path(f'{subjects_dir}', f'{subject}', 'mne_files')
        bem_check = f'{subjects_dir}/{subject}/bem/'
        eye_proj1 = f'{DATA_DIR}/{subject}_eyes1-proj.fif.gz'
        eye_proj2 = f'{DATA_DIR}/{subject}_eyes2-proj.fif.gz'
        fname_meg = f'{DATA_DIR}/{subject}_ses-rest_task-rest.fif'
        t1_fname = os.path.join(subjects_dir, subject, 'mri', 'T1.mgz')
        heartbeat_proj = f'{DATA_DIR}/{subject}_heartbeat-proj.fif.gz'
        fwd_fname = f'{DATA_DIR}/{subject}-fwd_{volume_spacing}.fif.gz'
        src_fname = f'{DATA_DIR}/{subject}-src_{volume_spacing}.fif.gz'
        cov_fname = f'{DATA_DIR}/{subject}-cov_{volume_spacing}.fif.gz'
        raw_proj = f'{DATA_DIR}/{subject}_ses-rest_task-rest_proj.fif.gz'
        source_voxel_coords = f'{DATA_DIR}/{subject}_coords_{volume_spacing}.pkl'
        corr_data_false_file_ac = f'{DATA_DIR}/{subject}_corr_ortho_false_{volume_spacing}_{frequency}_ac.npy'
        corr_data_true_file_ac = f'{DATA_DIR}/{subject}_corr_ortho_true_{volume_spacing}_{frequency}_ac.npy'
        corr_data_false_file_sc = f'{DATA_DIR}/{subject}_corr_ortho_false_{volume_spacing}_{frequency}_sc.npy'
        corr_data_true_file_sc = f'{DATA_DIR}/{subject}_corr_ortho_true_{volume_spacing}_{frequency}_sc.npy'
        corr_data_false_file_vc = f'{DATA_DIR}/{subject}_corr_ortho_false_{volume_spacing}_{frequency}_vc.npy'
        corr_data_true_file_vc = f'{DATA_DIR}/{subject}_corr_ortho_true_{volume_spacing}_{frequency}_vc.npy'
        trans = f'/home/senthil/Downloads/tmp/camcan_coreg-master/trans/{subject}-trans.fif' # The transformation file obtained by coregistration

        #compute_bem(subject, subjects_dir)
        #compute_scalp_surfaces(subject, subjects_dir)
        file_trans = pathlib.Path(trans)
        file_ss = pathlib.Path(src_fname)
        file_fm = pathlib.Path(fwd_fname)
        file_proj = pathlib.Path(raw_proj)
        file_cov = pathlib.Path(cov_fname)
        isdir_bem = pathlib.Path(bem_check)

        if not file_trans.exists():
            print (f'{trans} File doesnt exist...')
            coregistration(subject, subjects_dir, trans)

        info = mne.io.read_info(fname_meg)
        # plot_registration(info, trans, subject, subjects_dir)

        if not file_ss.exists():
            compute_SourceSpace(subject, subjects_dir, src_fname, source_voxel_coords, plot=True, ss=space, 
                                volume_spacing=volume_spacing)
        src = mne.read_source_spaces(src_fname)
        # view_SS_brain(subject, subjects_dir, src)
        if not file_fm.exists():
            forward_model(subject, subjects_dir, fname_meg, trans, src, fwd_fname)
        fwd = mne.read_forward_solution(fwd_fname)
        # sensitivty_plot(subject, subjects_dir, fwd)

        raw = mne.io.read_raw_fif(fname_meg, verbose='error', preload=True)
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

        if not file_cov.exists():
            cov = mne.compute_raw_covariance(raw_proj_applied) # compute before band-pass of interest
            mne.write_cov(cov_fname, cov)
        cov = mne.read_cov(cov_fname) 

        # cov.plot(raw.info, proj=True, exclude='bads', show_svd=False
        # raw_proj_applied.crop(tmax=10)
        raw_proj_applied.filter(l_freq=freqs[key][0], h_freq=freqs[key][1], n_jobs=16)
        events = mne.make_fixed_length_events(raw_proj_applied, duration=5.)
        epochs = mne.Epochs(raw_proj_applied, events=events, tmin=0, tmax=5.,
                            baseline=None, preload=True)
        # epochs = epochs.resample(5, npad='auto')
        # plot_psd(epochs)
        data_cov = mne.compute_covariance(epochs)

        '''
        Get the seed location from each subject T1 Image
        '''
        t1 = nib.load(t1_fname)
        vox_mri_t = t1.header.get_vox2ras_tkr()
        mri_vox_t = np.linalg.inv(vox_mri_t)
        sources = []
        for src_ in src:
            points = src_['rr'][src_['inuse'].astype(bool)]
            sources.append(apply_trans(mri_vox_t, points * 1e3))
            sources = np.concatenate(sources, axis=0)
        sources_vox = np.round(sources)
        sources_mni = source_to_MNI(subject, subjects_dir, t1, sources_vox)
        sources_mni = np.round(sources_mni)

        interest_left_ac = ROI_mni['AC_Left']
        interest_right_ac = ROI_mni['AC_Right']
        interest_left_sc = ROI_mni['SSC_Left']
        interest_right_sc = ROI_mni['SSC_Right']
        interest_left_vc = ROI_mni['VC_Left']
        interest_right_vc = ROI_mni['VC_Right']

        seed_left_ac = locate_seed(interest_left_ac, sources_mni)
        seed_right_ac = locate_seed(interest_right_ac, sources_mni)
        seed_left_sc = locate_seed(interest_left_sc, sources_mni)
        seed_right_sc = locate_seed(interest_right_sc, sources_mni)
        seed_left_vc = locate_seed(interest_left_vc, sources_mni)
        seed_right_vc = locate_seed(interest_right_vc, sources_mni)

        print(f'Left seed index Auditory cortex {seed_left_ac}') if seed_left_ac != 0  else sys.exit()
        print(f'Right seed index Auditory cortex {seed_right_ac}') if seed_right_ac != 0  else sys.exit()
        print(f'Left seed index Somatosensory cortex {seed_left_sc}') if seed_left_sc != 0  else sys.exit()
        print(f'Right seed index Somatosensory cortex {seed_right_sc}') if seed_right_sc != 0  else sys.exit()
        print(f'Left seed index Visual cortex {seed_left_vc}') if seed_left_vc != 0  else sys.exit()
        print(f'Right seed index Visual cortex {seed_right_vc}') if seed_right_vc != 0  else sys.exit()

        seed_l = []; seed_l.append(seed_left_ac); seed_l.append(seed_left_sc); seed_l.append(seed_left_vc)
        seed_r = []; seed_r.append(seed_right_ac); seed_r.append(seed_right_sc); seed_r.append(seed_right_vc)

        if space == 'volume':
            filters = make_lcmv(epochs.info, fwd, data_cov, 0.05, cov,
                            pick_ori='max-power', weight_norm='nai')
            epochs.apply_hilbert()
            stcs = apply_lcmv_epochs(epochs, filters, verbose=True, return_generator=True)
                
            # Power Envelope Correlation
            corr_false = False
            if corr_false:
                print(f'Computing Power Envelope Correlation for {subject}....Orthogonalize False')
                corr_false = envelope_correlation(stcs, verbose=True, orthogonalize=False, seed=seed_l, n_jobs=20)
                del stcs
            else:
                print(f'Computing Power Envelope Correlation for {subject}....Orthogonalize True')
                
                corr_true_ac = envelope_correlation(stcs, verbose=True, seed=seed_left_ac, n_jobs=18)
                np.save(corr_data_true_file_ac, corr_true_ac[seed_right_ac])
                print(f'Correlation between Left AC and Right AC: {corr_true_ac[seed_right_ac]}')

                stcs = apply_lcmv_epochs(epochs, filters, verbose=True, return_generator=True) # Re-setting generator
                corr_true_sc = envelope_correlation(stcs, verbose=True, seed=seed_left_sc, n_jobs=18)
                np.save(corr_data_true_file_sc, corr_true_sc[seed_right_sc])
                print(f'Correlation between Left SC and Right SC: {corr_true_sc[seed_right_sc]}')

                stcs = apply_lcmv_epochs(epochs, filters, verbose=True, return_generator=True) # Re-setting generator
                corr_true_vc = envelope_correlation(stcs, verbose=True, seed=seed_left_vc, n_jobs=18)
                np.save(corr_data_true_file_vc, corr_true_vc[seed_right_vc])
                print(f'Correlation between Left VC and Right VC: {corr_true_vc[seed_right_vc]}')
                
                del stcs

    time_elapsed = datetime.now() - start_t
    print ('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

# elif space == 'surface':
#     inv = make_inverse_operator(epochs.info, fwd, cov, loose=1.0)

#     # labels = mne.read_labels_from_annot(subject, 'aparc.a2009s',
#     #                                     subjects_dir=subjects_dir)
#     epochs.apply_hilbert()  # faster to apply in sensor space
#     stcs = apply_inverse_epochs(epochs, inv, lambda2=1. / 9., pick_ori='normal',
#                                 return_generator=False)
#     stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2=1. / 9., pick_ori='normal')
#     #stcs.save(stcs_fname)
#     #stcs = mne.read_source_estimate(stcs_fname, subject=subject)
#     # print(f'Source Estimate: {stcs}')
#     # np.save('/home/senthil/Downloads/tmp/stc.npy', stcs.data)

#     # label_ts = mne.extract_label_time_course(
#     #     stcs, labels, inv['src'], return_generator=True)
#     corr = envelope_correlation(stcs, verbose=True, orthogonalize=False)
#     np.save(f'{subjects_dir}/corr_ortho_false.npy', corr)

# let's plot this matrix
# from nilearn import datasets
# atlas = datasets.fetch_atlas_destrieux_2009()
# fig = plt.figure(figsize=(11,10))
# plt.imshow(corr, interpolation='None', cmap='RdYlBu_r')
# plt.yticks(range(len(atlas.labels)), atlas.labels[1:])
# plt.xticks(range(len(atlas.labels)), atlas.labels[1:], rotation=90)
# plt.title('Parcellation correlation matrix')
# plt.colorbar()
# plt.show()

# fig, ax = plt.subplots(figsize=(4, 4))
# im = ax.imshow(corr, cmap='viridis', clim=np.percentile(corr, [5, 95]))
# fig.tight_layout()
# fig.colorbar(im)
# plt.show()
# view_source_origin(corr, labels, inv, subjects_dir)
