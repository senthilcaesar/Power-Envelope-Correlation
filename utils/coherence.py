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


@timefn
def envelope_coherence(se_data, seed_l, seed_r, fmin, fmax):

        '''
        se_data == used adaptive linear spatial filtering (beamforming)
        to estimate the spectral amplitude and phase of neuronal signals at the source
        level   

        Example:
        seed_l = Index of Left somatosensory cortex source estimate data
        seed_r = Index of Right somatosensory cortex source estimate data
        '''
        se_data = se_data.data[[seed_l,seed_r]].copy()
        data_mag = np.abs(se_data)
    
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
            spectral_estimate = cwt(data_mag, wvlt)
            spectral_estimate = spectral_estimate[:,0,:]

            power_envelope = np.log(np.square(np.abs(spectral_estimate)))
            power_envelope = power_envelope[np.newaxis,:,:]

            coherency, freqs, times, n_epochs, n_tapers = spectral_connectivity(
                power_envelope, fmin=fmin, fmax=fmax, method='cohy',faverage=True, sfreq=sfreq, n_jobs=4)
            
            coherence_corr = np.real(coherency)
            coherence_corr = coherence_corr[1,:,:][0]
            covar_freq_list.append(coherence_corr)

        coherence_correlation = np.vstack(covar_freq_list)
        '''
        coherence_correlation.shape = (26,278)
        26 is the co-variation frequency (x-axis) [0.032 - 10]
        278 is the carrier freqeuncy (y-axis) [4 - 128]
        '''
        return coherence_correlation
