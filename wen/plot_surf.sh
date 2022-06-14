#!/bin/bash

freqs=(1 2 3 4 5 6)
min='0.0'
mid='0.17'
max='0.37'
name="cluster"
freeSurfer_dir='/Applications/freesurfer/7.1.1/bin'

for i in ${freqs[@]}; do

    freq=${i}
    surf_dir="/Applications/freesurfer/7.1.1/subjects/fsaverage/surf"
    sub_dir="/Users/senthilp/Desktop/wen/degree/correlation/${name}_mapped_vol_freq${freq}"

    lh_inflated="${sub_dir}/lh.${name}_mapped_array_freq${freq}.avgMapping_allSub_RF_ANTs_MNI152_orig_to_fsaverage.nii.gz"
    rh_inflated="${sub_dir}/rh.${name}_mapped_array_freq${freq}.avgMapping_allSub_RF_ANTs_MNI152_orig_to_fsaverage.nii.gz"

    ${freeSurfer_dir}/freeview -f ${surf_dir}/lh.inflated:overlay=${lh_inflated}:overlay_threshold=${min},${mid},${max}:overlay_color='colorwheel','inverse' \
    --camera azimuth 2 elevation 11 roll -17 --zoom 1.2 --screenshot "${freq}lh1.jpg"
    ${freeSurfer_dir}/freeview -f ${surf_dir}/lh.inflated:overlay=${lh_inflated}:overlay_threshold=${min},${mid},${max}:overlay_color='colorwheel','inverse' \
    --camera azimuth 180 elevation 0 roll 14 --zoom 1.2 --screenshot "${freq}lh2.jpg"
    ${freeSurfer_dir}/freeview -f ${surf_dir}/lh.inflated:overlay=${lh_inflated}:overlay_threshold=${min},${mid},${max}:overlay_color='colorwheel','inverse' \
    --camera azimuth 90 elevation 90 roll 0 --zoom 1.2 --screenshot "${freq}lh3.jpg"

    ${freeSurfer_dir}/freeview -f ${surf_dir}/rh.inflated:overlay=${rh_inflated}:overlay_threshold=${min},${mid},${max}:overlay_color='colorwheel','inverse' \
    --camera azimuth 2 elevation 11 roll -17 --zoom 1.2 --screenshot "${freq}rh1.jpg"
    ${freeSurfer_dir}/freeview -f ${surf_dir}/rh.inflated:overlay=${rh_inflated}:overlay_threshold=${min},${mid},${max}:overlay_color='colorwheel','inverse' \
    --camera azimuth 180 elevation 0 roll 14 --zoom 1.2 --screenshot "${freq}rh2.jpg"
    ${freeSurfer_dir}/freeview -f ${surf_dir}/rh.inflated:overlay=${rh_inflated}:overlay_threshold=${min},${mid},${max}:overlay_color='colorwheel','inverse' \
    --camera azimuth 90 elevation 90 roll 0 --zoom 1.2 --screenshot "${freq}rh3.jpg"

done
