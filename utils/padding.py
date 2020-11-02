import numpy as np
import mne
from mne import create_info, EpochsArray
from mne.io.pick import channel_type

def pad_along_axis(array, target_length, axis):

    pad_size = target_length - array.shape[axis]
    
    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (int(pad_size/2), int(pad_size/2))
    print('symmetric mirror padding the data...')
    return np.pad(array, pad_width=npad, mode='symmetric')


def symmetric_padding_epochs(raw_data, epochs):

    epochs_array = []
    for segment in epochs:
        x, y = segment.shape
        data = segment.reshape(x,y)
        data_pad = pad_along_axis(data, y*3, axis=1)
        epochs_array.append(data_pad)

    final = np.dstack(epochs_array).swapaxes(0,2).swapaxes(1,2)
    ch_names = raw_data.ch_names
    ch_types = [channel_type(raw_data.info, i) for i in range(0, len(ch_names)) ]
    sfreq = raw_data.info['sfreq']
            
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    epochs_pad = EpochsArray(data=final, info=info)
    return epochs_pad


