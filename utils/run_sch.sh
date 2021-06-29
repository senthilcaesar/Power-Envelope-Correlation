#!/bin/bash

py_bin='/home/senthilp/anaconda3/envs/mne/bin/python'
base_dir='/home/senthilp/caesar/Power-Envelope-Correlation/utils'

${py_bin} ${base_dir}/create_corr_vol.py 
${py_bin} ${base_dir}/regis_corr_vol.py
#${py_bin} ${base_dir}/avg_corr_vol.py
