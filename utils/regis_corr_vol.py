import numpy as np
import multiprocessing as mp
import subprocess

from scipy.signal.signaltools import _inputs_swap_needed


def non_linear_registration(case, freq, sensor):
    '''

    Usage:

    antsRegistrationSyN.sh -d ImageDimension -f FixedImage -m MovingImage -o OutputPrefix

     -d:  ImageDimension: 2 or 3 (for 2 or 3 dimensional registration of single volume)

     -f:  Fixed image(s) or source image(s) or reference image(s)

     -m:  Moving image(s) or target image(s)

     -o:  OutputPrefix: A prefix that is prepended to all output files.

     -n:  Number of threads

    '''
    
    subdir = '/home/senthil/caesar/camcan/cc700/freesurfer_output'
    fsaverage = '/home/senthil/caesar/camcan/cc700/freesurfer_output/fsaverage/mri'
    input_file = f'{subdir}/{case}/mne_files/{case}_true_7.8_{freq}_{sensor}_corr.nii.gz'
    output_file = f'{subdir}/{case}/mne_files/{case}_{freq}_{sensor}_ants'
    bash_cmd = f'antsRegistrationSyNQuick.sh -d 3 -f {fsaverage}/brain.mgz -m {input_file} -o {output_file} -n 4'
    print(bash_cmd)
    subprocess.check_output(bash_cmd, shell=True)


def apply_transform(case, freq, sensor):
    
    subdir = '/home/senthil/caesar/camcan/cc700/freesurfer_output'
    fsaverage = '/home/senthil/caesar/camcan/cc700/freesurfer_output/fsaverage/mri'
    input_file = f'{subdir}/{case}/mne_files/{case}_true_7.8_{freq}_{sensor}_corr.nii.gz'
    output_file = f'{subdir}/{case}/mne_files/{case}_{freq}_{sensor}_antsWarped.nii.gz'
    trans_file1 = f'{subdir}/{case}/mne_files/{case}_2_sc_ants1Warp.nii.gz'
    trans_file2 = f'{subdir}/{case}/mne_files/{case}_2_sc_ants0GenericAffine.mat'
    bash_cmd =  f'antsApplyTransforms -d 3 -i {input_file} -r {fsaverage}/brain.mgz -o {output_file} -t {trans_file1} -t {trans_file2}'
    print(bash_cmd)
    subprocess.check_output(bash_cmd, shell=True)


cases = '/home/senthil/caesar/camcan/cc700/freesurfer_output/50.txt'
with open(cases) as f:
     case_list = f.read().splitlines()

freq = 4
sensor = ['sc', 'ac', 'vc']
for label in sensor:
    pool = mp.Pool(processes=10)
    for index, subject in enumerate(case_list):
        pool.apply_async(apply_transform, args=[subject, freq, label])
    pool.close()
    pool.join()
