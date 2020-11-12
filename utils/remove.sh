#!/bin/bash

while read -r case
do
	rm ${case}/mne_files/*.npy
	rm ${case}/mne_files/*.nii.gz
	rm ${case}/mne_files/*.mat
	rm ${case}/mne_files/*.html
	rm ${case}/mne_files/*.pkl
	rm ${case}/mne_files/*.h5
	rm ${case}/mne_files/*.png
	rm ${case}/mne_files/*.fif.gz
	rm ${case}/mne_files/*_4_16
	rm ${case}/mne_files/*_30_16

done < 22.txt
