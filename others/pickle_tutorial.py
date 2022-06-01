
import nibabel as nib
import numpy as np

def _get_lut(fname=None):
    """Get a FreeSurfer LUT."""
    if fname is None:
        fname = f'/Applications/freesurfer/7.2.0/FreeSurferColorLUT.txt'

    dtype = [('id', '<i8'), ('name', 'U'),
             ('R', '<i8'), ('G', '<i8'), ('B', '<i8'), ('A', '<i8')]
    lut = {d[0]: list() for d in dtype}
    with open(fname, 'r') as fid:
        for line in fid:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            line = line.split()
            if len(line) != len(dtype):
                raise RuntimeError(f'LUT is improperly formatted: {fname}')
            for d, part in zip(dtype, line):
                lut[d[0]].append(part)
    lut = {d[0]: np.array(lut[d[0]], dtype=d[1]) for d in dtype}
    assert len(lut['name']) > 0
    return lut

def read_freesurfer_lut(fname=None):
    lut = _get_lut(fname)
    names, ids = lut['name'], lut['id']
    colors = np.array([lut['R'], lut['G'], lut['B'], lut['A']], float).T
    atlas_ids = dict(zip(names, ids))
    colors = dict(zip(names, colors))
    return atlas_ids, colors

mgz_fname = 'shen_freesurfer.mgz'
atlas = nib.load(mgz_fname)
data = np.asarray(atlas.dataobj)
want = np.unique(data)


atlas_ids, colors = read_freesurfer_lut()

# function to return key for any value
def get_key(val):
    for key, value in atlas_ids.items():
         if val == value:
             return key
 
    return "key doesn't exist"


keep = np.in1d(list(atlas_ids.values()), want)

keys = sorted((key for ki, key in enumerate(atlas_ids.keys()) if keep[ki]),
              key=lambda x: atlas_ids[x])
    
not_in = []
b = list(atlas_ids.values())
for a in want:
    a = int(a)
    if a not in b:
        not_in.append(a)

import pickle
with open('label_dict.pkl', 'rb') as handle:
    b = pickle.load(handle)
    
    
import pickle

# save dictionary to pickle file
with open('label_dict.pickle', 'wb') as file:
    pickle.dump(res, file, protocol=pickle.HIGHEST_PROTOCOL)
