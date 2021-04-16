import os
import nibabel as nib


def Average(lst):
    return sum(lst) / len(lst)

age = ['18to29_avg','30to39_avg','40to49_avg','50to59_avg','60to69_avg','70to79_avg','80to88_avg']
seed = 'tmpcRight'

for directory in age:
    corr = []
    for filename in os.listdir(directory):
        postfix = f'{seed}.nii.gz'
        if filename.endswith(postfix):
            #print(filename)
            input_file = os.path.join(directory, filename)
            img_data = nib.load(input_file).get_fdata()
            corr.append(img_data.max())

    print(f'{directory}-------------')
    print(max(corr))
    #print(Average(corr))
    print('-----------------------')
