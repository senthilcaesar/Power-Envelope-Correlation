import numpy as np
import networkx as nx
from pathlib import Path
import mne
import multiprocessing as mp
'''The NumPy array is interpreted as an adjacency matrix for the graph'''


def run_between_graph(subjects_dir, subject, volume_spacing, freq):

	frequency = str(freq)
	DATA_DIR = Path(f'{subjects_dir}', f'{subject}', 'mne_files')
	fname_aseg = f'{subjects_dir}/{subject}/mri/aparc.a2009s+aseg.mgz'
	fwd_fname = f'{DATA_DIR}/{subject}_{volume_spacing}-fwd-label.fif.gz'
	label_corr = f'{DATA_DIR}/{subject}_corr_ortho_true_{volume_spacing}_{frequency}-label.npy'
	between_mapped = f'{DATA_DIR}/{subject}_corr_ortho_true_{volume_spacing}_{frequency}-betweenmapped.npy'
	fwd = mne.read_forward_solution(fwd_fname)

	threshold_prop = 0.5
	connectivity = np.load(label_corr)[0,:,:]
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
	degree[degree > 0.0] = 1

	G = nx.from_numpy_array(degree, create_using=nx.DiGraph)
	# nx.draw(G, with_labels=True)
	btw = nx.betweenness_centrality(G, normalized=True)
	btw_lst = list(btw.values())
	btw_arr = np.array(btw_lst)

	# Map the betweeness 190 label values to source space
	stc = mne.labels_to_stc(fname_aseg, btw_arr, src=fwd['src'])
	btw_values = stc.data[:,0]
	print(f'Betweeness saved to {between_mapped}')
	np.save(between_mapped, btw_values)


freqs = [6, 8, 12, 16]
cases = '/home/senthilp/caesar/camcan/cc700/freesurfer_output/full.txt'
subjects_dir = '/home/senthilp/caesar/camcan/cc700/freesurfer_output'
with open(cases) as f:
     case_list = f.read().splitlines()


def main():
    volume_spacing = 7.8
    for freq in freqs:
        print(f'Data filtered at frequency {str(freq)} Hz...')
        pool = mp.Pool(processes=25)
        for subject in case_list:
            pool.apply_async(run_between_graph, args=[subjects_dir, subject, volume_spacing, freq])
        pool.close()
        pool.join()

if __name__ == "__main__":
    import time 
    startTime = time.time()
    main()
    print('The script took {0} second !'.format(time.time() - startTime))
