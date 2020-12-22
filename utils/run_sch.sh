#!/bin/bash

py_bin='/home/senthilp/anaconda3/envs/mnev2/bin'
base_dir='/home/senthilp/caesar/Power-Envelope-Correlation/utils'

${py_bin}/python ${base_dir}/create_corr_vol.py 
${py_bin}/python ${base_dir}/regis_corr_vol.py
${py_bin}/python ${base_dir}/avg_corr_vol.py
