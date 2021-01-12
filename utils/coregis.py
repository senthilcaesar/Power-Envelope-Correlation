import mne
import multiprocessing as mp
import os
import subprocess
from IPython.display import Image
from mayavi import mlab
from common import *
import pathlib
from mne.viz import plot_alignment, set_3d_view
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
    mlab.savefig('/home/senthilp/Desktop/coreg.jpg')
    Image(filename='/home/senthilp/Desktop/coreg.jpg', width=500)
    mlab.show()


def MEG_head_registration(case_list):
    for subject in case_list:
        trans = f'{coregis_dir}/{subject}-trans.fif' # The transformation file obtained by coregistration
        file_trans = pathlib.Path(trans)
        if not file_trans.exists():
            print(f'{trans} File doesnt exist... {subject}')
            coregistration(subject, subjects_dir, trans)


def bem_and_scalp(subject, subjects_dir):
    print(f'Processing subject {subject}...')
    set_num_threads(10)
    compute_bem(subject, subjects_dir)
    compute_scalp_surfaces(subject, subjects_dir)

cases = '/home/senthilp/caesar/camcan/cc700/freesurfer_output/31to41.txt'
subjects_dir = '/home/senthilp/caesar/camcan/cc700/freesurfer_output'
coregis_dir = '/home/senthilp/caesar/camcan/cc700/camcan_coreg-master/trans'
with open(cases) as f:
     case_list = f.read().splitlines()


pool = mp.Pool(processes=15)
for subject in case_list:
    pool.apply_async(bem_and_scalp, args=[subject, subjects_dir])
pool.close()
pool.join()
