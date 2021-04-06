#!/bin/bash

age_array=("18to29" "30to39" "40to49" "50to59" "60to69" "70to79" "80to88")
surf_dir="${SUBJECTS_DIR}/fsaverage/surf"

min='0.05'
mid='0.07'
max='0.09'
freq='16'
seed='mtRight'

for i in ${age_array[@]}; do
    age=${i}
    lh_inflated="lh.${age}_true_${freq}_${seed}.avgMapping_allSub_RF_ANTs_MNI152_orig_to_fsaverage.nii.gz"
    rh_inflated="rh.${age}_true_${freq}_${seed}.avgMapping_allSub_RF_ANTs_MNI152_orig_to_fsaverage.nii.gz"
    freeview -f ${surf_dir}/lh.inflated:overlay=${lh_inflated}:overlay_threshold=${min},${mid},${max} --camera azimuth 13 elevation 18 roll -59 --screenshot "${age}_${seed}-${freq}-lh.jpg" 2
    freeview -f ${surf_dir}/rh.inflated:overlay=${rh_inflated}:overlay_threshold=${min},${mid},${max} --camera azimuth -16 elevation -189 roll 69 --screenshot "${age}_${seed}-${freq}-rh.jpg" 2
done
