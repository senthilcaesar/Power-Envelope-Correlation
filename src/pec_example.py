import os.path as op

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.connectivity import envelope_correlation
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.preprocessing import compute_proj_ecg, compute_proj_eog

data_dir = '/home/senthil/Downloads'
data_path = mne.datasets.brainstorm.bst_resting.data_path()
subjects_dir = op.join(data_path, 'subjects')
subject = 'bst_resting'
trans = op.join(data_path, 'MEG', 'bst_resting', 'bst_resting-trans.fif')
src = op.join(subjects_dir, subject, 'bem', subject + '-oct-6-src.fif')
bem = op.join(subjects_dir, subject, 'bem', subject + '-5120-bem-sol.fif')
raw_fname = op.join(data_path, 'MEG', 'bst_resting',
                    'subj002_spontaneous_20111102_01_AUX.ds')

raw = mne.io.read_raw_ctf(raw_fname, verbose='error')
raw.crop(0, 60).load_data().pick_types(meg=True, eeg=False).resample(80)
raw.apply_gradient_compensation(3)
projs_ecg, _ = compute_proj_ecg(raw, n_grad=1, n_mag=2)
projs_eog, _ = compute_proj_eog(raw, n_grad=1, n_mag=2, ch_name='MLT31-4407')
raw.info['projs'] += projs_ecg
raw.info['projs'] += projs_eog
raw.apply_proj()
cov = mne.compute_raw_covariance(raw)  # compute before band-pass of interest

##############################################################################
# Now we band-pass filter our data and create epochs.

raw.filter(14, 30)
events = mne.make_fixed_length_events(raw, duration=10.)
epochs = mne.Epochs(raw, events=events, tmin=0, tmax=10.,
                    baseline=None, reject=dict(mag=8e-13), preload=True)

##############################################################################
# Compute the forward and inverse
# -------------------------------

src = mne.read_source_spaces(src)
fwd = mne.make_forward_solution(epochs.info, trans, src, bem)
inv = make_inverse_operator(epochs.info, fwd, cov)
del fwd, src

##############################################################################
# Compute label time series and do envelope correlation
# -----------------------------------------------------

labels = mne.read_labels_from_annot(subject, 'aparc.a2009s',
                                    subjects_dir=subjects_dir)

import pickle
datafile = f'{data_dir}/label_names.pkl'
F = open(datafile, 'wb')
pickle.dump(labels, F)
F.close()

#print(labels[104])
#print(labels[329])
print(len(labels))
epochs.apply_hilbert()  # faster to apply in sensor space
stcs = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2=1. / 9., pick_ori='vector')

stcs_fname = f'{data_dir}/fixed_ori'
stcs.save(stcs_fname)

label_ts = mne.extract_label_time_course(
    stcs, labels, inv['src'], mode='mean_flip', return_generator=False)

import pickle
datafile = f'{data_dir}/label_ts.pkl'
F = open(datafile, 'wb')
pickle.dump(label_ts, F)
F.close()
F = open(datafile, 'rb')
label_ts = pickle.load(F)

corr = envelope_correlation(stcs, orthogonalize=False)
np.save(f'{data_dir}/corr_ortho_false.npy', corr)

#let's plot this matrix
# fig, ax = plt.subplots(figsize=(4, 4))
# im = ax.imshow(corr, cmap='viridis', vmin=corr.min(), vmax=corr.max(), clim=np.percentile(corr, [5, 95]))
# fig.tight_layout()
# fig.colorbar(im)
# plt.show()

##############################################################################
# Compute the degree and plot it
# ------------------------------
# from mayavi import mlab
# threshold_prop = 0.15  # percentage of strongest edges to keep in the graph
# degree = mne.connectivity.degree(corr, threshold_prop=threshold_prop)
# np.save(f'{data_dir}/degree_ortho_false.npy', degree)
# stc = mne.labels_to_stc(labels, degree)
# stc = stc.in_label(mne.Label(inv['src'][0]['vertno'], hemi='lh') +
#                     mne.Label(inv['src'][1]['vertno'], hemi='rh'))

#print(len(inv['src'][0]['vertno']))
#print(inv['src'][1]['vertno'])

# np.save(f'{data_dir}/stc_ortho_false.npy', stc.data)
# brain = stc.plot(
#     clim=dict(kind='value', lims=[stc.data.min(), stc.data.mean(), stc.data.max()]), colormap='gnuplot',
#     subjects_dir=subjects_dir, views='dorsal', hemi='both',
#     smoothing_steps=25, time_label='Beta band')
# mlab.show()