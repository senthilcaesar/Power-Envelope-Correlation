import pandas as pd
import pandas
import seaborn
import matplotlib.pyplot as plt 
import numpy as np
from os import path

participants = '/home/senthilp/caesar/camcan/cc700/meg/pipeline/release004/BIDS_20190411/meg_rest_raw/participants.tsv'
df = pd.read_csv(participants, sep='\t')
one = np.array(df[['participant_id', 'age']])
freq = 16

lst_key = list(one[:,0])
lst_val = list(one[:,1])
lst_key = [subname.replace('sub-', '') for subname in lst_key]

res = dict(zip(lst_key, lst_val))

TOT_file=f'/home/senthilp/caesar/camcan/cognitive' \
f'/cc700-scored/TOT/release001/summary/TOT_summary.txt'
subject_dir = '/home/senthilp/caesar/camcan/cc700/freesurfer_output'

df_tot = pd.read_csv(TOT_file, delimiter="\t")
df_tot = df_tot[['Subject', 'ToT_ratio']]

sub_list = []
sub_tot_list = []
sub_age_list = []
sub_corr_SC_list = []
sub_corr_AC_list = []
sub_corr_VC_list = []
sub_corr_LPC_list = []
sub_corr_TMPC_list = []
sub_corr_DPFC_list = []
sub_corr_MT_list = []

for index, row in df_tot.iterrows():
    sub = row['Subject']
    sub_tot = row['ToT_ratio']
    
    if sub in res:
        corr1_sc = f'{subject_dir}/sub-{sub}/mne_files/sub-{sub}_corr_ortho_true_7.8_{freq}_scLeft.npy'
        corr2_sc = f'{subject_dir}/sub-{sub}/mne_files/sub-{sub}_corr_ortho_true_7.8_{freq}_scRight.npy'
        
        corr1_ac = f'{subject_dir}/sub-{sub}/mne_files/sub-{sub}_corr_ortho_true_7.8_{freq}_acLeft.npy'
        corr2_ac = f'{subject_dir}/sub-{sub}/mne_files/sub-{sub}_corr_ortho_true_7.8_{freq}_acRight.npy'
        
        corr1_vc = f'{subject_dir}/sub-{sub}/mne_files/sub-{sub}_corr_ortho_true_7.8_{freq}_vcLeft.npy'
        corr2_vc = f'{subject_dir}/sub-{sub}/mne_files/sub-{sub}_corr_ortho_true_7.8_{freq}_vcRight.npy'
        
        corr1_lpc = f'{subject_dir}/sub-{sub}/mne_files/sub-{sub}_corr_ortho_true_7.8_{freq}_lpcLeft.npy'
        corr2_lpc = f'{subject_dir}/sub-{sub}/mne_files/sub-{sub}_corr_ortho_true_7.8_{freq}_lpcRight.npy'
        
        corr1_tmpc = f'{subject_dir}/sub-{sub}/mne_files/sub-{sub}_corr_ortho_true_7.8_{freq}_tmpcLeft.npy'
        corr2_tmpc = f'{subject_dir}/sub-{sub}/mne_files/sub-{sub}_corr_ortho_true_7.8_{freq}_tmpcRight.npy'
        
        corr1_dpfc = f'{subject_dir}/sub-{sub}/mne_files/sub-{sub}_corr_ortho_true_7.8_{freq}_dpfcLeft.npy'
        corr2_dpfc = f'{subject_dir}/sub-{sub}/mne_files/sub-{sub}_corr_ortho_true_7.8_{freq}_dpfcRight.npy'
        
        corr1_mt = f'{subject_dir}/sub-{sub}/mne_files/sub-{sub}_corr_ortho_true_7.8_{freq}_mtLeft.npy'
        corr2_mt = f'{subject_dir}/sub-{sub}/mne_files/sub-{sub}_corr_ortho_true_7.8_{freq}_mtRight.npy'
        
        if path.exists(corr1_sc) and path.exists(corr2_sc):
            corr_1_sc = float(np.load(corr1_sc)[1]) * 1.73
            corr_2_sc = float(np.load(corr2_sc)[0]) * 1.73
            corr_1_ac = float(np.load(corr1_ac)[3]) * 1.73
            corr_2_ac = float(np.load(corr2_ac)[2]) * 1.73
            corr_1_vc = float(np.load(corr1_vc)[5]) * 1.73
            corr_2_vc = float(np.load(corr2_vc)[4]) * 1.73
            corr_1_mt = float(np.load(corr1_mt)[7]) * 1.73
            corr_2_mt = float(np.load(corr2_mt)[6]) * 1.73
            corr_1_lpc = float(np.load(corr1_lpc)[13]) * 1.73
            corr_2_lpc = float(np.load(corr2_lpc)[12]) * 1.73
            corr_1_dpfc = float(np.load(corr1_dpfc)[15]) * 1.73
            corr_2_dpfc = float(np.load(corr2_dpfc)[14]) * 1.73
            corr_1_tmpc = float(np.load(corr1_tmpc)[17]) * 1.73
            corr_2_tmpc = float(np.load(corr2_tmpc)[16]) * 1.73
            corr_val_sc = (corr_1_sc + corr_2_sc) / 2
            corr_val_ac = (corr_1_ac + corr_2_ac) / 2
            corr_val_vc = (corr_1_vc + corr_2_vc) / 2
            corr_val_lpc = (corr_1_lpc + corr_2_lpc) / 2
            corr_val_dpfc = (corr_1_dpfc + corr_2_dpfc) / 2
            corr_val_tmpc = (corr_1_tmpc + corr_2_tmpc) / 2
            corr_val_mt = (corr_1_mt + corr_2_mt) / 2
    
            sub_age = res[sub]
            sub_list.append(sub)
            sub_tot_list.append(sub_tot)
            sub_age_list.append(sub_age)
            sub_corr_SC_list.append(corr_val_sc)
            sub_corr_AC_list.append(corr_val_ac)
            sub_corr_VC_list.append(corr_val_vc)
            sub_corr_LPC_list.append(corr_val_lpc)
            sub_corr_DPFC_list.append(corr_val_dpfc)
            sub_corr_TMPC_list.append(corr_val_tmpc)
            sub_corr_MT_list.append(corr_val_mt)

output = f'/home/senthilp/caesar/Power-Envelope-Correlation/cognitive/output_{freq}.xlsx'
df_tot_final = pd.DataFrame(list(zip(sub_list, sub_tot_list, sub_age_list, sub_corr_SC_list, sub_corr_AC_list, sub_corr_VC_list,
                                     sub_corr_LPC_list, sub_corr_DPFC_list, sub_corr_TMPC_list, sub_corr_MT_list)), 
                  columns =['sub', 'tot', 'age', 'sc_corr', 'ac_corr', 'vc_corr', 'lpc_corr', 'dpfc_corr', 'tmpc_corr', 'mt_corr'])
df_tot_final = df_tot_final.sort_values(by=['age'])
df_tot_final.to_excel(output, index=False) 
