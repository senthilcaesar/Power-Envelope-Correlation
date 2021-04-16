#!/bin/bash

surf_dir="${SUBJECTS_DIR}/fsaverage/surf"
freqs=(2 3 4 6 8 12 16 24 32 48 64 96 128)
age_array=("18to29" "30to39" "40to49" "50to59" "60to69" "70to79" "80to88")
mapped_vol_dir="/Users/senthilp/Desktop/surface/output/mapped_vol"
max_arr=("0.07138496599509381" 
		 "0.08567332741222344" 
		 "0.09688612495665438" 
		 "0.11804669964476489" 
		 "0.12150360125815496" 
		 "0.12076978789991699" 
		 "0.1288041821680963")

min='0.05'
mid='0.07'
seed='tmpcRight'

k=0
for i in ${age_array[@]}; do
    age=${i}
    max=`echo "${max_arr[${k}]}"`
    snapshot_dir="/Users/senthilp/Desktop/surface/output/${age}"
    for freq in ${freqs[@]}; do
        lh_inflated="${mapped_vol_dir}/lh.${age}_true_${freq}_${seed}.avgMapping_allSub_RF_ANTs_MNI152_orig_to_fsaverage.nii.gz"
        rh_inflated="${mapped_vol_dir}/rh.${age}_true_${freq}_${seed}.avgMapping_allSub_RF_ANTs_MNI152_orig_to_fsaverage.nii.gz"
        freeview -f ${surf_dir}/lh.inflated:overlay=${lh_inflated}:overlay_threshold=${min},${mid},${max} --colorscale \
        --camera azimuth 13 elevation 18 roll -59 --screenshot "${snapshot_dir}/${age}_${seed}-${freq}-lh.jpg" 2
        freeview -f ${surf_dir}/rh.inflated:overlay=${rh_inflated}:overlay_threshold=${min},${mid},${max} --colorscale \
        --camera azimuth -16 elevation -189 roll 69 --screenshot "${snapshot_dir}/${age}_${seed}-${freq}-rh.jpg" 2
        done
    k=$((k+1))
done
