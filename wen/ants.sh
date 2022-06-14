#!/bin/bash


antsRegistrationSyNQuick.sh -d 3 -f /Applications/freesurfer/7.1.1/subjects/fsaverage/mri/brain.mgz -m shen_greyscale.nii.gz -o shenTofreesurfer_tranformation -n 8
antsApplyTransforms -d 3 -i shen_1mm_268_parcellation.nii.gz -r /Applications/freesurfer/7.1.1/subjects/fsaverage/mri/brain.mgz -o shen_freesurferSpace.nii.gz -t shenTofreesurfer_tranformation1Warp.nii.gz -t shenTofreesurfer_tranformation0GenericAffine.mat --interpolation NearestNeighbor
