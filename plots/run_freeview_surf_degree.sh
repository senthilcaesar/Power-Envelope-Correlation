#!/bin/bash

surf_dir="${SUBJECTS_DIR}/fsaverage/surf"
#freqs=(2 3 4 6 8 12 16 24 32 48 64 96 128)
freqs=(12)
age_array=("18to29" "30to39" "40to49" "50to59" "60to69" "70to79" "80to88")
mapped_vol_dir="/Users/senthilp/Desktop/surface/output/mapped_vol"
min='30'
mid='60'
seed='degreeMapped'
max_arr=("106" 
 		 "122" 
 		 "126" 
 		 "132" 
 		 "134" 
 		 "133" 
 		 "134")

k=0
for i in ${age_array[@]}; do
    age=${i}
    max=`echo "${max_arr[${k}]}"`
    snapshot_dir="/Users/senthilp/Desktop/surface/output/${age}"
    for freq in ${freqs[@]}; do
        lh_inflated="${mapped_vol_dir}/lh.${age}_${seed}_7.8_${freq}.avgMapping_allSub_RF_ANTs_MNI152_orig_to_fsaverage.nii.gz"
        rh_inflated="${mapped_vol_dir}/rh.${age}_${seed}_7.8_${freq}.avgMapping_allSub_RF_ANTs_MNI152_orig_to_fsaverage.nii.gz"
        freeview -f ${surf_dir}/lh.pial:overlay=${lh_inflated}:overlay_threshold=${min},${mid},${max} \
        --camera azimuth 0 elevation 8 roll -10 --zoom 1.0 --screenshot "${snapshot_dir}/${age}_${seed}-${freq}-lh1.jpg" 2
        freeview -f ${surf_dir}/lh.pial:overlay=${lh_inflated}:overlay_threshold=${min},${mid},${max} \
        --camera azimuth 180 elevation 0 roll 14 --zoom 1.0 --screenshot "${snapshot_dir}/${age}_${seed}-${freq}-lh2.jpg" 2
        freeview -f ${surf_dir}/lh.pial:overlay=${lh_inflated}:overlay_threshold=${min},${mid},${max} \
        --camera azimuth 90 elevation 90 roll 0 --zoom 1.0 --screenshot "${snapshot_dir}/${age}_${seed}-${freq}-lh3.jpg" 2
        freeview -f ${surf_dir}/rh.pial:overlay=${rh_inflated}:overlay_threshold=${min},${mid},${max} \
        --camera azimuth 0 elevation 8 roll -10 --zoom 1.0 --screenshot "${snapshot_dir}/${age}_${seed}-${freq}-rh1.jpg" 2
        freeview -f ${surf_dir}/rh.pial:overlay=${rh_inflated}:overlay_threshold=${min},${mid},${max} \
        --camera azimuth 180 elevation 0 roll 14 --zoom 1.0 --screenshot "${snapshot_dir}/${age}_${seed}-${freq}-rh2.jpg" 2
        freeview -f ${surf_dir}/rh.pial:overlay=${rh_inflated}:overlay_threshold=${min},${mid},${max} \
        --camera azimuth 90 elevation 90 roll 0 --zoom 1.0 --screenshot "${snapshot_dir}/${age}_${seed}-${freq}-rh3.jpg" 2

        done
    k=$((k+1))
done
