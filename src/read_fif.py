import mne
import numpy as np

def general_info(raw):
    print('--------------------------General Info-----------------------------')
    n_time_samps = raw.n_times
    time_secs = raw.times
    ch_names = raw.ch_names
    n_chan = len(ch_names)  # note: there is no raw.n_channels attribute
    print(f'The data object has {n_time_samps} time samples and {n_chan} channels.')
    print(f'The last time sample is at {time_secs[-1]} seconds.')
    print(f'The first few channel names are {(ch_names[:3])}.')
    # some examples of raw.info:
    print('bad channels:', raw.info['bads'])  # chs marked "bad" during acquisition
    print(raw.info['sfreq'], 'Hz')            # sampling frequency
    print(raw.info['description'])      # miscellaneous acquisition info
    print(f'Time 1 sec as index: {raw.time_as_index(1)}')
    print(f'Time 1,2,3 sec as index: {raw.time_as_index([1, 2, 3])}')
    print(f'Difference between 1,2,3 sec as index {np.diff(raw.time_as_index([1, 2, 3]))}')
    print('-------------------------------------------------------------------')

cases_f = '/home/senthil/caesar/camcan/cc700/meg/pipeline/release004/BIDS_20190411/meg_rest_raw/cases.txt'
with open(cases_f) as f:
    case_list = f.read().splitlines()

fname = case_list[50]
raw = mne.io.read_raw_fif(fname, verbose='error')
general_info(raw)


