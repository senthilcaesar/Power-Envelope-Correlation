import numpy as np

threshold_prop = 0.101
connectivity=np.load('/Users/senthilp/Desktop/PEC/corr_true.npy')

connectivity = np.array(connectivity)
n_nodes = len(connectivity)

if np.allclose(connectivity, connectivity.T):
    split = 2.
    connectivity[np.tril_indices(n_nodes)] = 0
else:
    split = 1.

threshold_prop = float(threshold_prop)
degree = connectivity.ravel()

degree[::n_nodes + 1] = 0.

n_keep = int(round((degree.size - len(connectivity)) *
                        threshold_prop / split))

degree[np.argsort(degree)[:-n_keep]] = 0
degree.shape = connectivity.shape
degree += degree.T
degree[degree > 0.05] = 1
degree_count = np.sum(degree > 0, axis=0)
