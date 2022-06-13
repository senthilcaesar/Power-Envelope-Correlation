age="80"
surf_dir="/Applications/freesurfer/7.1.1/subjects/fsaverage/surf"
sub_dir="/Users/senthilp/Desktop/Wen/degree/mapped_vol_${age}"

lh_inflated="${sub_dir}/lh.degree_mapped_array.avgMapping_allSub_RF_ANTs_MNI152_orig_to_fsaverage.nii.gz"
rh_inflated="${sub_dir}/rh.degree_mapped_array.avgMapping_allSub_RF_ANTs_MNI152_orig_to_fsaverage.nii.gz"

min='100'
mid='105'
max='110'

/Applications/freesurfer/7.1.1/bin/freeview -f ${surf_dir}/lh.inflated:overlay=${lh_inflated}:overlay_threshold=${min},${mid},${max} --colorscale \
--camera azimuth 21 elevation 15 roll -13 --zoom 1.5 --screenshot "${age}lh.jpg"

/Applications/freesurfer/7.1.1/bin/freeview -f ${surf_dir}/rh.inflated:overlay=${rh_inflated}:overlay_threshold=${min},${mid},${max} --colorscale \
--camera azimuth 161 elevation 13 roll 16 --zoom 1.5 --screenshot "${age}rh.jpg"
