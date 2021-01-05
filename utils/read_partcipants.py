import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

participants = '/home/senthil/caesar/camcan/cc700/mri' \
'/pipeline/release004/BIDS_20190411/anat/participants.tsv'

output = '/home/senthil/caesar/camcan/cc700/mri' \
'/pipeline/release004/BIDS_20190411/anat/participants_sorted.csv'

df = pd.read_csv(participants, sep='\t')
df_sort = df.sort_values(by=['age','gender_code'])
df_sort.to_csv(output, sep='\t')


data_dir='/home/senthil/caesar/camcan/cc700'

dataset1 = df[df['age'].between(18, 30)]
dataset2 = df[df['age'].between(42, 58)]
dataset3 = df[df['age'].between(68, 88)]
dataset4 = df[df['age'].between(31, 41)]
dataset5 = df[df['age'].between(59, 67)]

#dataset1['participant_id'].to_csv(f'{data_dir}/18to30.txt',index=False, sep=',',header=False)
#dataset2['participant_id'].to_csv(f'{data_dir}/42to58.txt',index=False, sep=',',header=False)
#dataset3['participant_id'].to_csv(f'{data_dir}/68to88.txt',index=False, sep=',',header=False)
dataset4['participant_id'].to_csv(f'{data_dir}/31to41.txt',index=False, sep=',',header=False)
dataset5['participant_id'].to_csv(f'{data_dir}/59to67.txt',index=False, sep=',',header=False)

# fig, ax = plt.subplots(figsize=(8,4))
# df['age'].plot.hist(bins=70)
# ax.set_yticks([0,5,10,15,20])
# ax.set_xlabel('Age', fontsize=8)
# ax.set_ylabel('Frequency', fontsize=8)
# ax.set_title(f'Total 653 Subjects (Cam-CAN MEG resting state dataset)', fontsize=8)
# plt.savefig('/home/senthil/Desktop/agerange.png', dpi=600)