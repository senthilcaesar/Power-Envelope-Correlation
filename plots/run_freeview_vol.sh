#!/bin/bash

freqs=(2 3 4 6 8 12 16 24 32 48 64 96 128)
age_array=("18to29" "30to39" "40to49" "50to59" "60to69" "70to79" "80to88")
seed='vcRight'
fsaverage="/Users/senthilp/Desktop/average"
freeview_bin="/Applications/freesurfer/7.1.1/bin"

scLeft_slice='170 74 102'
scRight_slice='90 80 96'
acLeft_slice='182 118 106'
acRight_slice='76 116 104'
vcLeft_slice='148 110 42'
vcRight_slice='112 102 48'
mt+Left_slice='-47 -69 -3'
mt+Right_slice='54 -63 -8'
mtlLeft_slice='-20 -40 -10'
mtlRight_slice='40 -40 0'
smcLeft_slice='-40 -40 -60'
smcRight_slice='40 -30 50'
lpcLeft_slice='-39 -54 32'
lpcRight_slice='46 -45 39'
dpfcLeft_slice='-40 30 50'
dpfcRight_slice='30 20 30'
tmpcLeft_slice='-50 -40 -10'
tmpcRight_slice='60 -20 0'
mpfcMidBrain_slice='-3 39 -2'
smaMidBrain_slice='-2 1 51'

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
max='0.105'
opacity='0.6'
colormap='nih'

for i in ${age_array[@]}; do
    age=${i}
    subjs_avg="/Users/senthilp/Desktop/average/${age}_avg"
    snapshot_dir="/Users/senthilp/Desktop/results/${age}"
    for freq in ${freqs[@]}; do
        ${freeview_bin}/freeview --layout 1 -v ${fsaverage}/T1.mgz \
        ${subjs_avg}/${age}_true_${freq}_${seed}.nii.gz:colormap=${colormap}:opacity=${opacity}:colorscale=${min},${max} \
        -slice ${coord} -colorscale \
        -viewport ${view} \
        --screenshot "${snapshot_dir}/${age}_${seed}${freq}.jpg" 2
    done
done
