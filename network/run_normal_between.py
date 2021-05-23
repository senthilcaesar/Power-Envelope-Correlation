import numpy as np

def Average(lst):
    return sum(lst) / len(lst)

degree = np.load('sub-CC110033_corr_ortho_true_7.8_16-degreemapped.npy')
between = np.load('sub-CC110033_corr_ortho_true_7.8_16-betweenmapped.npy')
between_normal_lst = []


for i, x in enumerate(between):
    
    some = np.argwhere(degree == degree[i])
    some = some[:,0]
    lst_mean = []
    lst_std = []
    
    for y in range(20):
        random_btw = np.random.choice(some, 15)
        btw_rand = np.take(between, random_btw)
    
        mean_btw = np.mean(btw_rand)
        std_btw = np.std(btw_rand)
        
        lst_mean.append(mean_btw)
        lst_std.append(std_btw)
    
    mean_avg = Average(lst_mean)
    mean_std = Average(lst_std)
    
    if std_btw > 0.0:    
        between_normal_val = (between[i] - mean_avg) / mean_std
    else:
        between_normal_val = (between[i] - mean_avg)
    between_normal_lst.append(between_normal_val)

between_normal = np.array(between_normal_lst)
