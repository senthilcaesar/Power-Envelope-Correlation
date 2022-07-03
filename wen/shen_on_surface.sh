# 1. Map shen_freesurfer.mgz volume to freesurfer fsaverage surface using (regfusion)
# 2. Volume to label ( mri_cor2label --c lh.shen_mapped_array.avgMapping_allSub_RF_ANTs_MNI152_orig_to_fsaverage.nii.gz --id 1 --l MF.label --surf fsaverage lh )


freesurfer_dir='/Applications/freesurfer/7.1.1/subjects/fsaverage'
freeview_bin='/Applications/freesurfer/7.1.1/bin'


${freeview_bin}/freeview -f ${freesurfer_dir}/surf/lh.inflated::curvature_method=binary::curvature_setting=-0.673989:label=${freesurfer_dir}/label/MF.label:label_color=#4682b4:label_outline=1:label=${freesurfer_dir}/label/FP.label:label_color=#f5f5f5:label_outline=1:label=${freesurfer_dir}/label/DMN.label:label_color=#cd3e4e:label_outline=1:label=${freesurfer_dir}/label/MOT.label:label_color=#781286:label_outline=1:label=${freesurfer_dir}/label/VI.label:label_color=#f27efa:label_outline=1:label=${freesurfer_dir}/label/VII.label:label_color=#46f2f4:label_outline=1:label=${freesurfer_dir}/label/VAs.label:label_color=#dcf8a4:label_outline=1:label=${freesurfer_dir}/label/SAL.label:label_color=#e69422:label_outline=1:label=${freesurfer_dir}/label/SC.label:label_color=#fcff2b:label_outline=1:label=${freesurfer_dir}/label/CBL.label:label_color=#00760e:label_outline=1
