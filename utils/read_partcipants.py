import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

participants = '/home/senthilp/caesar/camcan/cc700/meg/pipeline/release004/BIDS_20190411/meg_rest_raw/participants.tsv'

output = '/home/senthilp/caesar/camcan/cc700/meg/pipeline/release004/BIDS_20190411/meg_rest_raw/participants_sorted.csv'

df = pd.read_csv(participants, sep='\t')
df_sort = df.sort_values(by=['age','gender_code'])
df_sort.to_csv(output, sep='\t')


data_dir='/home/senthilp/caesar/camcan/cc700'

dataset1 = df[df['age'].between(18, 29)]
dataset2 = df[df['age'].between(30, 39)]
dataset3 = df[df['age'].between(40, 49)]
dataset4 = df[df['age'].between(50, 59)]
dataset5 = df[df['age'].between(60, 69)]
dataset6 = df[df['age'].between(70, 79)]
dataset7 = df[df['age'].between(80, 88)]

dataset1['participant_id'].to_csv(f'{data_dir}/18to29.txt',index=False, sep=',',header=False)
dataset2['participant_id'].to_csv(f'{data_dir}/30to39.txt',index=False, sep=',',header=False)
dataset3['participant_id'].to_csv(f'{data_dir}/40to49.txt',index=False, sep=',',header=False)
dataset4['participant_id'].to_csv(f'{data_dir}/50to59.txt',index=False, sep=',',header=False)
dataset5['participant_id'].to_csv(f'{data_dir}/60to69.txt',index=False, sep=',',header=False)
dataset6['participant_id'].to_csv(f'{data_dir}/70to79.txt',index=False, sep=',',header=False)
dataset7['participant_id'].to_csv(f'{data_dir}/80to88.txt',index=False, sep=',',header=False)

# fig, ax = plt.subplots(figsize=(8,4))
# df['age'].plot.hist(bins=70)
# ax.set_yticks([0,5,10,15,20])
# ax.set_xlabel('Age', fontsize=8)
# ax.set_ylabel('Frequency', fontsize=8)
# ax.set_title(f'Total 653 Subjects (Cam-CAN MEG resting state dataset)', fontsize=8)
# plt.savefig('/home/senthilp/Desktop/agerange.png', dpi=600)
