import pickle
import mne

#F_open = open('label_ts.pkl', 'rb')
#rest_data = pickle.load(F_open)


src_space_fname = '/home/senthil/Downloads/tmp/sub-CC221373-src.fif.gz'
# src_space_fname = '/home/senthil/mne_data/MNE-brainstorm-data/bst_resting' \
#                     '/subjects/bst_resting/bem/bst_resting-oct-6-src.fif'
src_space = mne.read_source_spaces(src_space_fname)
lh_surf_coord = src_space[0]['rr']     # Triangle Mesh coordinates
lh_triangle_idx = src_space[0]['tris'] # traingular mesh face of 3 vertices


#src_tcs_fname = '/home/senthil/Downloads/fixed_ori-lh.stc'

tcs_fname = '/home/senthil/Downloads/tmp/sub-CC221373-fixed_ori-lh.stc'
tcs = mne.read_source_estimate(tcs_fname)
tcs_data = tcs.data # Both hemisphere data
'''
 vertex indices ( every index value for vertices will
 select a coordinate from lh_surf_coord )
 
 lh_data source data is mapped to lh_ver_idx which is then mapped to lh_surf_coord
 
 Each triangle lh_triangle_idx consist of 3 vertices
 
'''
lh_ver_idx = tcs.vertices[0]
lh_data = tcs.lh_data # Left hemipshere data