import numpy as np
import multiprocessing as mp
from multiprocessing import Semaphore
from ..filter import next_fast_len
import math
from ..source_estimate import _BaseSourceEstimate
from ..utils import verbose, _check_combine, _check_option
from multiprocessing import Process, Manager, Value, Pool
from mne.connectivity import spectral_connectivity

def compute_coherence(epoch_data, corrs):

        combine='mean'
        orthogonalize="pairwise"
        verbose=None
        epoch_data = epoch_data.data
        n_nodes, n_times = epoch_data.shape
        data_mag = np.abs(epoch_data)

        # Logarithm of the squared amplitude envelope ( Power Envelope ) From Hipps paper
        data_mag = np.log(np.square(data_mag))


        #log_range = np.arange(-1.5,1.1,0.1)
        #center_freq = [math.pow(10,val) for val in log_range]
        '''
        We chose a spectral bandwidth of 0.95 octaves (f/ σ f ~3.15) and spaced the center frequencies log-
        arithmically according to the exponentiation of the base 10 with exponents ranging from −1.5 in steps of 0.1

        We derived spectral estimates in successive half-overlapping temporal windows that cov-
        ered ±3 σ t . From these complex numbers, we derived the coherency between power envelopes and 
        took the real part of coherency as the frequency-specific measure of correlation

        '''
        center_freq = 0.032
        sfreq = 1000
        sigma = center_freq / 3.15
        wvlt = mne.time_frequency.morlet(sfreq, [center_freq], sigma=sigma)
        spectral_estimate = mne.time_frequency.tfr.cwt(data_mag, wvlt)

        power_envelope = np.log(np.square(np.abs(spectral_estimate)))

        coherency, freqs, times, n_epochs, n_tapers = spectral_connectivity(
            power_envelope, method='cohy',sfreq=sfreq, n_jobs=4)

        coherence_correlation = np.real(coherency)
        print(coherence_correlation)

        corrs.append(coherence_correlation)


@verbose
def envelope_coherence(data, combine='mean',verbose=None, seed=None, n_jobs=16):


    from scipy.signal import hilbert
    n_nodes = None
    if combine is not None:
        fun = _check_combine(combine, valid=('mean',))
    else:  # None
        fun = np.array

    pool = mp.Pool(processes=n_jobs)
    manager = mp.Manager()
    corrs = manager.list()
    for ei, epoch_data in enumerate(data):
        pool.apply_async(compute_coherence, args=[epoch_data, corrs])
    pool.close()
    pool.join()

    corrs = list(corrs)
    corr = fun(corrs)
    return corr
