import nibabel as nib


surf_img_file = '/home/senthil/anaconda3/lib/python3.7/site-packages/nilearn' \
                '/datasets/data/fsaverage5/pial_inflated.left.gii'

time_series = '/home/senthil/nilearn_data/nki_enhanced_surface/A00028185/A00028185_left_preprocessed_fwhm6.gii'

surf_img = nib.load(surf_img_file)
series_gii = surf_img.agg_data() # vertex-by-timestep array
coords_gii = surf_img.agg_data('pointset')
triangles_gii = surf_img.agg_data('triangle')