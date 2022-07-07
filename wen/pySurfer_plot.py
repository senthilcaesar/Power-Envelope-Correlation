import os
from surfer import Brain
from mayavi import mlab
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.image as mpimg


brain = Brain("fsaverage", "lh", "inflated", cortex=(0.5, 0.5, 0.5), 
              background='white', units='m',size=(500,500))
brain.show_view(view='lateral')
# (one of ‘lateral’, ‘medial’, ‘rostral’, ‘caudal’, ‘dorsal’, ‘ventral’, ‘frontal’, ‘parietal’)

overlay_file = "/Users/senthilp/Desktop/wen/degree/\
lh.degree_mapped_array_freq16.avgMapping_allSub_RF_ANTs_MNI152_orig_to_fsaverage.nii.gz"

labels = ["MF", "FP", "DMN", "MOT", "VI", "VII", "VAs", "SAL", "SC", "CBL"]
colors = ['#4682b4', '#f5f5f5', '#cd3e4e', '#781286', '#f27efa', '#46f2f4', '#dcf8a4', '#e69422', '#fcff2b', '#00760e']

# Add labels to brain surface
for i in range(len(labels)):
    brain.add_label(labels[i], borders=True, hemi='lh', color=colors[i])
    
brain.add_overlay(overlay_file, min=0.35, max=0.4, name="overlay_file") # Change MIN, MAX value here
brain.overlays["overlay_file"].pos_bar.lut_mode = "autumn"
mlab.savefig('/Users/senthilp/Desktop/degree_avg.jpg',size=(2000, 2000))
mlab.close(all=True)

# Plot label legends
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.figsize'] = (8, 7)
plt.rcParams['savefig.dpi'] = 600
img = mpimg.imread('/Users/senthilp/Desktop/degree_avg.jpg')
elements = [Line2D([0], [0], label = l, color = c) for l, c in zip(labels, colors)]
plt.imshow(img)
leg = plt.legend(handles = elements, fontsize=12, title='Shen Network', bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0)
for line in leg.get_lines():
    line.set_linewidth(3.0)
frame = leg.get_frame()
frame.set_color('lightgray')
plt.gca().set_axis_off()
plt.savefig("/Users/senthilp/Desktop/degree_avg_legend.jpg")
