from nilearn import datasets
import matplotlib as mpl
import matplotlib.pyplot as plt

nki_dataset = datasets.fetch_surf_nki_enhanced(n_subjects=1)

print(('Resting state data of the first subjects on the '
       'fsaverag5 surface left hemisphere is at: %s' %
      nki_dataset['func_left'][0]))

# Destrieux parcellation for left hemisphere in fsaverage5 space
destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
parcellation = destrieux_atlas['map_left']
labels = destrieux_atlas['labels']

# Fsaverage5 surface template
fsaverage = datasets.fetch_surf_fsaverage()

# The fsaverage dataset contains file names pointing to
# the file locations
print('Fsaverage5 pial surface of left hemisphere is at: %s' %
      fsaverage['pial_left'])
print('Fsaverage5 inflated surface of left hemisphere is at: %s' %
      fsaverage['infl_left'])
print('Fsaverage5 sulcal depth map of left hemisphere is at: %s' %
      fsaverage['sulc_left'])


from nilearn import surface

timeseries = surface.load_surf_data(nki_dataset['func_left'][0])
pcc_region = b'G_cingul-Post-dorsal'
import numpy as np
pcc_labels = np.where(parcellation == labels.index(pcc_region))[0]
seed_timeseries = np.mean(timeseries[pcc_labels], axis=0)

from scipy import stats

stat_map = np.zeros(timeseries.shape[0])
for i in range(timeseries.shape[0]):
    stat_map[i] = stats.pearsonr(seed_timeseries, timeseries[i])[0]

# Re-mask previously masked nodes (medial wall)
stat_map[np.where(np.mean(timeseries, axis=1) == 0)] = 0

# Transform ROI indices in ROI map
pcc_map = np.zeros(parcellation.shape[0], dtype=int)
pcc_map[4294] = 1
pcc_map[4295] = 1

corr_flag = 'true'
a = np.load(f'/home/senthil/Downloads/tmp/corr_ortho_{corr_flag}.npy')
b = a[4294,:][:10240]
b = np.pad(b, (0, 2), 'constant')

from nilearn import plotting
display = plotting.plot_surf_roi(fsaverage['infl_left'], roi_map=pcc_map,
                        hemi='left', view='medial',
                        darkness=0.6, bg_map=fsaverage['sulc_left'],
                        title='Seed')
display.savefig(f'/home/senthil/Desktop/1_{corr_flag}.png', dpi=1200)

display = plotting.plot_surf_stat_map(fsaverage['infl_left'], stat_map=b,
                            hemi='left', view='medial', colorbar=False,
                            bg_map=fsaverage['sulc_left'], bg_on_data=True,
                            darkness=.3, title='Correlation map', vmax=b.max(),
                            cmap=plotting.cm.bwr)


norm = mpl.colors.Normalize(vmin=b.min(),vmax=b.max())
sm = plt.cm.ScalarMappable(cmap=plotting.cm.bwr, norm=norm)
sm.set_array([])

cax = display.add_axes([0.87, 0.3, 0.01, 0.5])
display.colorbar(sm, ticks=np.linspace(b.min(),b.max(),5), cax=cax)
display.savefig(f'/home/senthil/Desktop/2_{corr_flag}.png', dpi=1200)

display = plotting.plot_surf_stat_map(fsaverage['infl_left'], stat_map=b,
                            hemi='left', view='lateral', colorbar=False,
                            bg_map=fsaverage['sulc_left'], bg_on_data=True,
                            darkness=.3, title='Correlation map', vmax=b.max(),
                            cmap=plotting.cm.bwr)

cax = display.add_axes([0.87, 0.3, 0.01, 0.5])
display.colorbar(sm, ticks=np.linspace(b.min(),b.max(),5), cax=cax)
display.savefig(f'/home/senthil/Desktop/3_{corr_flag}.png', dpi=1200)
