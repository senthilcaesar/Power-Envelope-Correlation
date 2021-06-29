import numpy as np
import multiprocessing as mp
import subprocess
from settings import Settings


def non_linear_registration(case, freq, label, spacing):
    '''

    Usage:

    antsRegistrationSyN.sh -d ImageDimension -f FixedImage -m MovingImage -o OutputPrefix

     -d:  ImageDimension: 2 or 3 (for 2 or 3 dimensional registration of single volume)

     -f:  Fixed image(s) or source image(s) or reference image(s)

     -m:  Moving image(s) or target image(s)

     -o:  OutputPrefix: A prefix that is prepended to all output files.

     -n:  Number of threads

    '''
    
    subdir = '/home/senthilp/caesar/camcan/cc700/freesurfer_output'
    fsaverage = '/home/senthilp/freesurfer/subjects/fsaverage/mri'
    input_file = f'{subdir}/{case}/mne_files/{case}_true_{spacing}_{freq}_{sensor}_corr.nii.gz'
    output_file = f'{subdir}/{case}/mne_files/{case}_ants'
    bash_cmd = f'antsRegistrationSyNQuick.sh -d 3 -f {fsaverage}/brain.mgz -m {input_file} -o {output_file} -n 4'
    print(bash_cmd)
    subprocess.check_output(bash_cmd, shell=True)


def apply_transform(case, freq, label, spacing):

    '''
    antsApplyTransforms: Usage

     -d, --dimensionality 2/3/4
          This option forces the image to be treated as a specified-dimensional image. If 
          not specified, antsWarp tries to infer the dimensionality from the input image. 

     -i, --input inputFileName
          Currently, the only input objects supported are image objects. However, the 
          current framework allows for warping of other objects such as meshes and point 
          sets. 

     -r, --reference-image imageFileName
          For warping input images, the reference image defines the spacing, origin, size, 
          and direction of the output warped image. 

     -o, --output warpedOutputFileName

     -t, --transform transformFileName

     '''
    flag = str(flag)
    subdir = f'/home/senthilp/caesar/camcan/cc700/freesurfer_output'
    fsaverage = f'/home/senthilp/freesurfer/subjects/fsaverage/mri'
    input_file = f'{subdir}/{case}/mne_files/{case}_{label}_{spacing}_{freq}.nii.gz'
    output_file = f'{subdir}/{case}/mne_files/{case}_{label}_{spacing}_{freq}_antsWarped.nii.gz'
    trans_file_warp = f'{subdir}/trans/{case}_ants1Warp.nii.gz'
    trans_file_rigid = f'{subdir}/trans/{case}_ants0GenericAffine.mat'

    bash_cmd =  f'antsApplyTransforms -d 3 -i {input_file} -r {fsaverage}/brain.mgz -o {output_file} -t {trans_file_warp} -t {trans_file_rigid} --interpolation NearestNeighbor'

    print(bash_cmd)
    subprocess.check_output(bash_cmd, shell=True)


if __name__ == '__main__':

    cases = '/home/senthilp/caesar/camcan/cc700/freesurfer_output/full.txt'
    subjects_dir = '/home/senthilp/caesar/camcan/cc700/freesurfer_output'
    spacing = '7.8'
    freqs = [12,]
    sensor = ['degreeMapped',]
    njobs = 20

    
    with open(cases) as f:
        case_list = f.read().splitlines()
    for freq in freqs:
        for label in sensor:
            pool = mp.Pool(processes=njobs)
            for index, subject in enumerate(case_list):
                #pool.apply_async(non_linear_registration, args=[subject, freq, label, spacing])
                pool.apply_async(apply_transform, args=[subject, freq, label, spacing])
            pool.close()
            pool.join()
