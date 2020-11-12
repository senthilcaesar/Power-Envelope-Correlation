#!/bin/bash

freqs=(2 3 4 6 8 12 16 24 32 48 64 96 128)

scLeft_slice='170 74 102'
scRight_slice='90 80 96'


seed='scLeft'

i=1
for freq in ${freqs[@]}; do
	
	/usr/local/freesurfer/bin/freeview --layout 1 -v fsaverage/mri/T1.mgz \
	average/72_true_${freq}_${seed}.nii.gz:colormap=nih:opacity=0.6:colorscale=0.0,0.09 \
	-slice ${scLeft_slice} -colorscale \
	-viewport coronal \
	--screenshot "/home/senthil/Desktop/screenshots/${seed}${freq}.jpg" 2
	i=$((i+1))
done
