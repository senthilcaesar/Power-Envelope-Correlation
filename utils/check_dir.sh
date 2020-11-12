#!/bin/bash

while read -r case
do
	check_f="/home/senthil/caesar/camcan/cc700/freesurfer_output/${case}/mne_files/${case}_true_7.8_48_scLeft_corr.nii.gz"
	[ ! -f ${check_f} ] && echo ${case} #echo "${check_f} DOES NOT exists."

done < 18to30.txt
