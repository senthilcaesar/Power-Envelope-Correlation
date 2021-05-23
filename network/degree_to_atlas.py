import numpy as np
from pathlib import Path
import mne
import multiprocessing as mp



def run_degreeMapping(subjects_dir, subject, volume_spacing, freq):
    
    sensor = 'degreemapped'
    frequency = str(freq)
    DATA_DIR = Path(f'{subjects_dir}', f'{subject}', 'mne_files')
    fwd_fname = f'{DATA_DIR}/{subject}_{volume_spacing}-fwd-label.fif.gz'
    label_corr = f'{DATA_DIR}/{subject}_corr_ortho_true_{volume_spacing}_{frequency}-label.npy'
    degree_mapped = f'{DATA_DIR}/{subject}_corr_ortho_true_{volume_spacing}_{frequency}-{sensor}.npy'
    
    corr = np.load(label_corr)
    corr = corr[0, :, :]
    fwd = mne.read_forward_solution(fwd_fname)
    fname_aseg = f'{subjects_dir}/{subject}/mri/aparc.a2009s+aseg.mgz'
    threshold_prop = 0.15
    degree = mne.connectivity.degree(corr, threshold_prop=threshold_prop)

	stc = mne.labels_to_stc(fname_aseg, degree, src=fwd['src'])
	degree_values = stc.data[:,0]
	print(f'Degree saved to {degree_mapped}')
	np.save(degree_mapped, degree_values)


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
            pool.apply_async(run_degreeMapping, args=[subjects_dir, subject, volume_spacing, freq])
        pool.close()
        pool.join()

if __name__ == "__main__":
    import time 
    startTime = time.time()
    main()
    print('The script took {0} second !'.format(time.time() - startTime))
