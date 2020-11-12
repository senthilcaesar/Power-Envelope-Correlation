#!/bin/bash

filename='22.txt'
echo "Total cases = `cat $filename | wc -l`"
echo

i=1
while read -r case
do
	#mv ${case}/mne_files/${case}_ants1Warp.nii.gz trans/
	mv ${case}/mne_files/${case}_ants0GenericAffine.mat trans/
	echo "Case ${i} done"
	i=$((i+1))

done < ${filename}
