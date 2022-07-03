#!/bin/bash

: <<'END_COMMENT'
1) Map shen_freesurfer.mgz volume to freesurfer fsaverage surface using (regfusion)
2) Volume to label ( mri_cor2label --c lh.shen_mapped_array.avgMapping_allSub_RF_ANTs_MNI152_orig_to_fsaverage.nii.gz --id 1 --l lh_MF.label --surf fsaverage lh )
1  =    [MF]
2  =    [FP]
3  =   [DMN]
4  =   [Mot]
5  =    [VI]
6  =   [VII]
7  =   [VAs]
8  =   [SAL]
9  =    [SC]
10 =   [CBL]
END_COMMENT

freesurfer_dir='/usr/local/freesurfer/7.2.0/subjects/fsaverage'
freeview_bin='/usr/local/freesurfer/7.2.0/bin'
hemi='lh'

${freeview_bin}/freeview -f ${freesurfer_dir}/surf/${hemi}.inflated:curvature_method=binary:curvature_setting=-0.673989:label=${freesurfer_dir}/label/${hemi}_MF.label:label_color=#4682b4:label_outline=1:label=${freesurfer_dir}/label/${hemi}_FP.label:label_color=#f5f5f5:label_outline=1:label=${freesurfer_dir}/label/${hemi}_DMN.label:label_color=#cd3e4e:label_outline=1:label=${freesurfer_dir}/label/${hemi}_MOT.label:label_color=#781286:label_outline=1:label=${freesurfer_dir}/label/${hemi}_VI.label:label_color=#f27efa:label_outline=1:label=${freesurfer_dir}/label/${hemi}_VII.label:label_color=#46f2f4:label_outline=1:label=${freesurfer_dir}/label/${hemi}_VAs.label:label_color=#dcf8a4:label_outline=1:label=${freesurfer_dir}/label/${hemi}_SAL.label:label_color=#e69422:label_outline=1:label=${freesurfer_dir}/label/${hemi}_SC.label:label_color=#fcff2b:label_outline=1:label=${freesurfer_dir}/label/${hemi}_CBL.label:label_color=#00760e:label_outline=1
