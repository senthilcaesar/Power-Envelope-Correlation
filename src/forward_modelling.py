import mne
import numpy as np
import os
import subprocess
import numpy as np
from surfer import Brain
from IPython.display import Image
from mayavi import mlab
from mne.viz import plot_alignment, set_3d_view
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from mne.connectivity import envelope_correlation
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
import matplotlib.pyplot as plt
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


def coregistration(subject, subjects_dir):
    mne.gui.coregistration(subject=subject, subjects_dir=subjects_dir)


def plot_registration(info, trans, subject, subjects_dir):
    fig = plot_alignment(info, trans, subject=subject, dig=True,
                        meg=True, subjects_dir=subjects_dir,
                        coord_frame='head')
    set_3d_view(figure=fig, azimuth=135, elevation=80)
    mlab.savefig('/Users/senthilp/Desktop/coreg.jpg')
    Image(filename='/Users/senthilp/Desktop/coreg.jpg', width=500)
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


def compute_SourceSpace(subject, subjects_dir, src_fname, plot=True):
    src = mne.setup_source_space(subject, spacing='oct6', add_dist='patch',
                                subjects_dir=subjects_dir)
    src.save(src_fname, overwrite=True)
    if plot:
        mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                        src=src, orientation='coronal')
    return src


def forward_model(subject, subjects_dir, fname_meg, trans, src, fwd_fname):
    conductivity = (0.3, 0.006, 0.3)  # for three layers
    model = mne.make_bem_model(subject=subject, ico=4,
                            conductivity=conductivity,
                            subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    fwd = mne.make_forward_solution(fname_meg, trans=trans, src=src, bem=bem,
                                    meg=True, eeg=False, mindist=5.0, n_jobs=2)
    print(fwd)
    mne.write_forward_solution(fwd_fname, fwd, overwrite=True, verbose=None)
    leadfield = fwd['sol']['data']
    print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)


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
    brain = stc.plot(
        clim=dict(kind='percent', lims=[75, 85, 95]), colormap='gnuplot',
        subjects_dir=subjects_dir, views='dorsal', hemi='both',
        smoothing_steps=25, time_label='Beta band')
    mlab.show()


# cases_meg = '/home/senthil/caesar/camcan/cc700/meg/pipeline/release004/BIDS_20190411/meg_rest_raw/cases.txt'
# cases_T1 = '/home/senthil/caesar/camcan/cc700/mri/pipeline/release004/BIDS_20190411/anat/cases.txt'
# with open(cases_meg) as f:
#     case_meg_list = f.read().splitlines()
# with open(cases_T1) as f:
#     cases_T1_list = f.read().splitlines()

# fname_meg = case_meg_list[120]
# fname_T1 = cases_T1_list[120]
# print(fname_meg, fname_T1)

subject='sub-CC221373'
subjects_dir='/Users/senthilp/Downloads/tmp'
fname_meg = f'{subjects_dir}/{subject}_ses-rest_task-rest.fif'
fwd_fname = f'{subjects_dir}/{subject}-fwd.fif.gz'
src_fname = f'{subjects_dir}/{subject}-src.fif.gz'
cov_fname = f'{subjects_dir}/{subject}-cov.fif.gz'

# compute_bem(subject, subjects_dir)
# compute_scalp_surfaces(subject, subjects_dir)
# coregistration(subject, subjects_dir)

# The transformation file obtained by coregistration
trans = f'{subjects_dir}/{subject}-trans.fif'
info = mne.io.read_info(fname_meg)

# plot_registration(info, trans, subject, subjects_dir)
# compute_SourceSpace(subject, subjects_dir, src_fname, plot=Falseafari)
src = mne.read_source_spaces(src_fname)
# view_SS_brain(subject, subjects_dir, src)
# forward_model(subject, subjects_dir, fname_meg, trans, src, fwd_fname)
fwd = mne.read_forward_solution(fwd_fname)
#sensitivty_plot(subject, subjects_dir, fwd)

raw = mne.io.read_raw_fif(fname_meg, verbose='error')
# projs_ecg, _ = compute_proj_ecg(raw, n_grad=1, n_mag=2)
# projs_eog, _ = compute_proj_eog(raw, n_grad=1, n_mag=2)
# raw.info['projs'] += projs_ecg
# raw.info['projs'] += projs_eog
# raw.apply_proj()
# cov_arr = mne.compute_raw_covariance(raw)
# cov_arr.save(cov_fname)
cov = mne.read_cov(cov_fname)
events = mne.find_events(raw, stim_channel=['STI201','STI101','STI003'], 
                        uint_cast=True, initial_event=True, verbose=False)
epochs = mne.Epochs(raw, events=events,
                    baseline=None, preload=True, event_repeated='merge')
inv = make_inverse_operator(epochs.info, fwd, cov)

labels = mne.read_labels_from_annot(subject, 'aparc',
                                    subjects_dir=subjects_dir)
epochs.apply_hilbert()  # faster to apply in sensor space
stcs = apply_inverse_epochs(epochs, inv, lambda2=1. / 9., pick_ori='normal',
                            return_generator=True)
label_ts = mne.extract_label_time_course(
    stcs, labels, inv['src'], return_generator=True)
corr = envelope_correlation(label_ts, verbose=True)

# let's plot this matrix
# fig, ax = plt.subplots(figsize=(4, 4))
# ax.imshow(corr, cmap='viridis', clim=np.percentile(corr, [5, 95]))
# fig.tight_layout()
# plt.show()
view_source_origin(corr, labels, inv, subjects_dir)