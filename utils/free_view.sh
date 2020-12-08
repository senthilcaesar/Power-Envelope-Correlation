#!/bin/bash

#freqs=(2 3 4 6 8 12 16 24 32 48 64 96 128)
freqs=(80.0 108.0 136.0 164.0 192.0 220.0)

fsaverage='/home/senthilp/freesurfer/subjects/fsaverage/mri'
freeview_bin='/home/senthilp/freesurfer/bin'
subjs_avg='/home/senthilp/caesar/camcan/cc700/freesurfer_output/average'
snapshot_dir='/home/senthilp/Desktop/screenshots'

scLeft_slice='170 74 102'
scRight_slice='90 80 96'
acLeft_slice='182 118 106'
acRight_slice='76 116 104'
vcLeft_slice='148 110 42'
vcRight_slice='112 102 48'

seed='vcRight'

if [ ${seed} = 'scLeft' ]; then
	coord=${scLeft_slice}
	view='coronal'
elif [ ${seed} = 'acLeft' ]; then
	coord=${acLeft_slice}
	view='axial'
elif [ ${seed} = 'vcLeft' ]; then
	coord=${vcLeft_slice}
	view='axial'
elif [ ${seed} = 'scRight' ]; then
	coord=${scRight_slice}
	view='coronal'
elif [ ${seed} = 'acRight' ]; then
	coord=${acRight_slice}
	view='axial'
elif [ ${seed} = 'vcRight' ]; then
	coord=${vcRight_slice}
	view='axial'
else
	coord='None'
fi


min='0.0'
max='0.09'
opacity='0.6'
colormap='nih'

echo ${coord}
for freq in ${freqs[@]}; do
	
	${freeview_bin}/freeview --layout 1 -v ${fsaverage}/T1.mgz \
	${subjs_avg}/72_true_${freq}_${seed}.nii.gz:colormap=${colormap}:opacity=${opacity}:colorscale=${min},${max} \
	-slice ${coord} -colorscale \
	-viewport ${view} \
	--screenshot "${snapshot_dir}/${seed}${freq}.jpg" 2

done
