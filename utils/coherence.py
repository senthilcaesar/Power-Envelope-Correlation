import numpy as np
import mne
import multiprocessing as mp
from ..utils import _check_combine
from mne.connectivity import spectral_connectivity
from mne.time_frequency import morlet
from mne.time_frequency.tfr import cwt
from functools import wraps
import time
import math
from padding import pad_along_axis

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


def compute_coherence(epoch_data, corrs, seed_l, seed_r):

        '''
        epoch_data == used adaptive linear spatial filtering (beamforming)
        to estimate the spectral amplitude and phase of neuronal signals at the source
        level   

        Example:
        seed_l = Left somatosensory cortex signal data
        seed_r = Right somatosensory cortex signal data
        '''
        padding = False
        epoch_data = epoch_data.data[[seed_l,seed_r]].copy()
        data_mag = np.abs(epoch_data)
        
        if padding:
            x, y = data_mag.shape
            data_mag_pad = pad_along_axis(data_mag, y*3, axis=1)
        else:
            data_mag_pad = data_mag

        log_range = np.arange(-1.5,1.1,0.1)
        covar_freqs = [math.pow(10,val) for val in log_range]
        '''
        We chose a spectral bandwidth of (σf = f * 3.15) and spaced the center frequencies log-
        arithmically according to the exponentiation of the base 10 with exponents ranging from −1.5 in steps of 0.1

        We derived spectral estimates in successive half-overlapping temporal windows that cov-
        ered ±3 σ t . From these complex numbers, we derived the coherency between power envelopes and 
        took the real part of coherency as the frequency-specific measure of correlation

        '''
        covar_freq_list = []
        sfreq = 1000
        for freq in covar_freqs:
            sigma = 3.15 * freq
            wvlt = morlet(sfreq, [freq], sigma=sigma, n_cycles=7)
            spectral_estimate = cwt(data_mag_pad, wvlt)
            spectral_estimate = spectral_estimate[:,0,:]

            power_envelope = np.abs(spectral_estimate)
            x, y = power_envelope.shape
            power_envelope = power_envelope.reshape(1,x,y)

            coherency, freqs, times, n_epochs, n_tapers = spectral_connectivity(
                power_envelope, fskip=250, fmin=4, fmax=128, method='cohy',sfreq=sfreq, n_jobs=16)

            np.save('/home/senthil/Downloads/freq.npy', freqs)
            
            coherence_corr = np.real(coherency)
            coherence_corr = coherence_corr[1,:,:][0]
            covar_freq_list.append(coherence_corr)

        coherence_correlation = np.vstack(freq_list)
        '''
        coherence_correlation.shape = (26,278)
        26 is the co-variation frequency (x-axis) [0.032 - 10]
        278 is the carrier freqeuncy (y-axis) [4 - 128]
        '''
        corrs.append(coherence_correlation)


@timefn
def envelope_coherence(data, combine='mean',verbose=None, seed_l=None, seed_r=None, n_jobs=2):
        
    if combine is not None:
        fun = _check_combine(combine, valid=('mean',))
    else:  # None
        fun = np.array

    pool = mp.Pool(processes=n_jobs)
    manager = mp.Manager()
    corrs = manager.list()
    for ei, epoch_data in enumerate(data):
        pool.apply_async(compute_coherence, args=[epoch_data, corrs, seed_l, seed_r])
    pool.close()
    pool.join()

    corrs = list(corrs)
    corr = fun(corrs)
    return corr