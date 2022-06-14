import os
import time
import numpy as np
import nibabel as nib
from scipy.io import loadmat
from regfusion import vol_to_fsaverage
from functools import wraps
import multiprocessing as mp

freq_lst = []
names = ['degree', 'between', 'path', 'cluster']
sub_dir = '/home/senthilp/caesar/wen/degree'
input_path = '/usr/local/freesurfer/7.2.0/subjects/fsaverage/mri/brain.mgz'
atlas = '/home/senthilp/caesar/wen/degree/atlas/shen_freesurferSpace.nii.gz'

for name in names:
    fname = f'Graph_{name}.mat'
    corr = loadmat(fname)
    freq_lst += list(corr[name])
    
'''
freq_lst is structured as below

freq_lst[0] ---> degree correlation of freq 4
freq_lst[1] ---> degree correlation of freq 8
freq_lst[2] ---> degree correlation of freq 12
freq_lst[3] ---> degree correlation of freq 16
freq_lst[4] ---> degree correlation of freq 20
freq_lst[5] ---> degree correlation of freq 24

freq_lst[6] ---> between correlation of freq 4
freq_lst[7] ---> between correlation of freq 8
freq_lst[8] ---> between correlation of freq 12
freq_lst[9] ---> between correlation of freq 16
freq_lst[10] ---> between correlation of freq 20
freq_lst[11] ---> between correlation of freq 24

freq_lst[12] ---> path correlation of freq 4
freq_lst[13] ---> path correlation of freq 8
freq_lst[14] ---> path correlation of freq 12
freq_lst[15] ---> path correlation of freq 16
freq_lst[16] ---> path correlation of freq 20
freq_lst[17] ---> path correlation of freq 24

freq_lst[18] ---> cluster correlation of freq 4
freq_lst[19] ---> cluster correlation of freq 8
freq_lst[20] ---> cluster correlation of freq 12
freq_lst[21] ---> cluster correlation of freq 16
freq_lst[22] ---> cluster correlation of freq 20
freq_lst[23] ---> cluster correlation of freq 24

'''

name_list = ['degree']*6 + ['between']*6 + ['path']*6 + ['cluster']*6
freq_names = [*range(4,28,4), *range(4,28,4), *range(4,28,4), *range(4,28,4)]

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


def num_threads(nt):
    nt = str(nt)
    os.environ["OMP_NUM_THREADS"] = nt         # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"] = nt    # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"] = nt         # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = nt  # export VECLIB_MAXIMUM_THREADS=1
    os.environ["NUMEXPR_NUM_THREADS"] = nt     # export NUMEXPR_NUM_THREADS=1


def mat_surf(freq, k, sub_dir, input_path, atlas, name):
    num_threads(6)
    print(f'Processing freq{k} {name}...')
    obj = nib.load(atlas)
    data = nib.load(atlas).get_fdata()
    want = np.arange(1,269)
    map_data = np.zeros((data.shape))
    
    for i, id in enumerate(want):
        index = np.where(data == id)
        map_data[index] = freq[i]
    
    output_file = f'{sub_dir}/{name}_mapped_array_freq{k}.nii.gz'
    t1 = nib.load(input_path)
    affine = t1.affine
    hdr = t1.header
    result_img = nib.Nifti1Image(map_data, affine, header=hdr)
    result_img.to_filename(output_file)
 
    # Map volume to surface
    lh, rh = vol_to_fsaverage(output_file, f'{name}_mapped_vol_freq{k}')


@timefn
def main():
    pool = mp.Pool(processes=len(freq_lst))   
    for k, freq in enumerate(freq_lst):
        pool.apply_async(mat_surf, args=[freq, freq_names[k], sub_dir, input_path, atlas, name_list[k]])
    pool.close()
    pool.join()
    
if __name__ == "__main__":
    main()
