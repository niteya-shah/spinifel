import h5py
import sys
import numpy as np
import random
from pathlib import Path


fnames = sys.argv[1:]
h5files = [h5py.File(fname, 'r') for fname in fnames]
n_files = len(h5files)


n_images_per_file = [h5file['intensities'].shape[0] for h5file in h5files]
n_total_images = np.sum(n_images_per_file)


# Create a random no. identifier for each image in all datasets
ids = list(range(n_total_images))
rand_ids = []
while ids:
    my_id = random.sample(ids, 1)[0]
    rand_ids += [my_id]
    ids.remove(my_id)
assert len(rand_ids) == np.unique(rand_ids).shape[0]


# Combine intensities and orientations
intensities = np.empty([n_total_images]+list(h5files[0]['intensities'].shape[1:]), dtype=h5files[0]['intensities'].dtype)
orientations = np.empty([n_total_images]+list(h5files[0]['orientations'].shape[1:]), dtype=h5files[0]['orientations'].dtype)
st = 0
for i, h5file in enumerate(h5files):
    print(f'copying st:{st} en:{st+n_images_per_file[i]}')
    intensities[st:st+n_images_per_file[i]] = h5file['intensities'][:]
    orientations[st:st+n_images_per_file[i]] = h5file['orientations'][:]
    st = n_images_per_file[i]


out_file = 'out.h5'
unique_keys = ['beam_offsets', 'fluences', 'pixel_index_map', 'pixel_position_reciprocal', 'polarization', 'solid_angle']
with h5py.File(out_file, 'w') as hf:
    # These are the same for all conformations
    for unique_key in unique_keys:
        hf.create_dataset(unique_key, data=h5files[0][unique_key][:])

    # Copy all volumes with new key and combine all command_lines
    command_line = ''
    for i, h5file in enumerate(h5files):
        hf.create_dataset(f'volume_{Path(fnames[i]).stem}', data=h5file['volume'][:])
        command_line += h5file.attrs['command_line'] + ';'

    # Combine intensities and orientations in random orders
    hf.create_dataset('intensities', data=intensities[rand_ids])
    hf.create_dataset('orientations', data=orientations[rand_ids])

    hf.attrs['command_line'] = command_line


# Check the output files
with h5py.File(out_file, 'r') as hf:
    for key in hf.keys():
        print(f'{key} {hf[key].shape}')
        if key in unique_keys:
            assert np.array_equal(hf[key][:], h5files[0][key][:])

    for i in range(3):
        assert np.array_equal(hf['intensities'][i], intensities[rand_ids[i]])





