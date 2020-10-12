import os.path as op
import mne
from mne.datasets import sample
data_path = sample.data_path()

# the raw file containing the channel location + types
print(data_path)
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
# The paths to Freesurfer reconstructions
subjects_dir = data_path + '/subjects'
subject = 'sample'

surf_fname = '/home/senthil/Downloads/surf_space.fif.gz'
vol_fname = '/home/senthil/Downloads/vol_space.fif.gz'


src = mne.setup_source_space(subject, spacing='ico5', add_dist='patch',
                             subjects_dir=subjects_dir)

src.save(surf_fname, overwrite=True)

surface = op.join(subjects_dir, subject, 'bem', 'inner_skull.surf')
vol_src = mne.setup_volume_source_space(subject, subjects_dir=subjects_dir,
                                        surface=surface)

vol_src.save(vol_fname, overwrite=True)
