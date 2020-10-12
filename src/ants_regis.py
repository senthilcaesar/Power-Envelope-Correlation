import numpy as np
import multiprocessing as mp
import subprocess


def non_linear_registration(case, ortho='true'):
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
    fsdir = '/home/senthil/mne_data/MNE-sample-data/subjects/fsaverage/mri'
    bash_cmd = (f'antsRegistrationSyNQuick.sh -d 3 -f {fsdir}/brain.mgz -m {subdir}/{case}/mne_files/' \
                f'{case}_corr_vol_{ortho}_5_no_morph_epoch.nii.gz -o {subdir}/{case}/mne_files/{case}_ants_orthoTrue_epoch -n 4')
    print(bash_cmd)
    subprocess.check_output(bash_cmd, shell=True)


cases = '/home/senthil/caesar/camcan/cc700/freesurfer_output/50.txt'
with open(cases) as f:
     case_list = f.read().splitlines()

pool = mp.Pool(processes=10)
manager = mp.Manager()
data_vse = manager.list()
for index, subject in enumerate(case_list):
    pool.apply_async(non_linear_registration, args=[subject, 'true'])
pool.close()
pool.join()
