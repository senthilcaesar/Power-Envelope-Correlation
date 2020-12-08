import numpy as np
import multiprocessing as mp
from ..utils import verbose, _check_combine, _check_option
from functools import wraps
import time

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


def compute_correlation(epoch_data, corrs, seed, orthogonalize):

        orthogonalize="pairwise"
        epoch_data = epoch_data.data.copy()
        n_nodes, n_times = epoch_data.shape
        #print(f'No of source points {n_nodes}') 
        #print(f'No of time points {n_times}')
        data_mag = np.abs(epoch_data)
        data_conj_scaled = epoch_data.conj()
        data_conj_scaled /= data_mag
        # subtract means
        data_mag_nomean = data_mag - np.mean(data_mag, axis=-1, keepdims=True)
        # compute variances using linalg.norm (square, sum, sqrt) since mean=0
        data_mag_std = np.linalg.norm(data_mag_nomean, axis=-1)
        data_mag_std[data_mag_std == 0] = 1

        corr = np.empty((n_nodes, 1))
        label_data = epoch_data[seed]

        if orthogonalize is False:
            label_data_orth = data_mag
            label_data_orth_std = data_mag_std

        else:
            label_data_orth = (label_data * data_conj_scaled).imag
            label_data_orth -= np.mean(label_data_orth, axis=-1,
                                           keepdims=True)
            label_data_orth_std = np.linalg.norm(label_data_orth, axis=-1)
            label_data_orth_std[label_data_orth_std == 0] = 1

        corr = np.dot(label_data_orth, data_mag_nomean[seed])
        corr /= data_mag_std[seed]
        corr /= label_data_orth_std

        if orthogonalize is not False:
            # Make it symmetric (it isn't at this point)
            corr = np.abs(corr)
            corr = (corr.T + corr) / 2.
        corrs.append(corr)
        del corr
        del epoch_data


@timefn
def envelope_correlation(data, combine='mean', orthogonalize="pairwise",
                         verbose=None, seed=None, n_jobs=10):


    _check_option('orthogonalize', orthogonalize, (False, 'pairwise'))
    from scipy.signal import hilbert
    n_nodes = None
    if combine is not None:
        fun = _check_combine(combine, valid=('mean',))
    else:  # None
        fun = np.array

    pool = mp.Pool(processes=n_jobs)
    manager = mp.Manager()
    corrs = manager.list()
    for epoch_data in data:
        pool.apply_async(compute_correlation, args=[epoch_data, corrs, seed, orthogonalize])
    pool.close()
    pool.join()

    corrs = list(corrs)
    corr = fun(corrs)
    return corr