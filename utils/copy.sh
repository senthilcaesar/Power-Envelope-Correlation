#!/bin/bash

filename='59to67.txt'
echo "Total cases = `cat $filename | wc -l`"
echo

i=1
while read -r case
do
	check_dir="/home/senthil/caesar/camcan/cc700/meg/pipeline/release004/BIDS_20190411/meg_rest_raw/${case}/ses-rest/meg/${case}_ses-rest_task-rest.fif"
	mv ${check_dir} ${case}/mne_files
	echo "Case ${i} done"
	i=$((i+1))

done < ${filename}
