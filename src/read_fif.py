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


def plot_event(events):
    event_dict = {'auditory/left': 256, 'auditory/right': 768, 'visual/left': 1792,
              'visual/right': 3840, 'smiley': 7936}

    fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'],
                              first_samp=raw.first_samp, event_id=event_dict)
    fig.subplots_adjust(right=0.7)  # make room for legend

    raw.plot(events=events, start=5, duration=10, color='gray',
             event_color={256: 'r', 768: 'g', 1792: 'b', 3840: 'm', 7936: 'y'}, block=True)


cases_f = '/home/senthil/caesar/camcan/cc700/meg/pipeline/release004/BIDS_20190411/meg_rest_raw/cases.txt'
with open(cases_f) as f:
    case_list = f.read().splitlines()

fname = case_list[120]
print(fname)
raw = mne.io.read_raw_fif(fname, verbose='error')
general_info(raw)
#raw.plot(block=True)
events = mne.find_events(raw, stim_channel=['STI201','STI101','STI003'], uint_cast=True, initial_event=True, verbose=False)
print(f"Shape of STIM events numpy array {np.shape(events)}")
print("Event array = (Index of event, Length of the event, Event type)")
print(f"STIM Event IDs: {np.unique(events[:,2])}\n")
print(events)
#plot_event(events)


