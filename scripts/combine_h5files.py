import h5py
import sys
import numpy as np
import random
from pathlib import Path
import time


t0 = time.monotonic()
fnames = sys.argv[1:]
print(f'Start opening {len(sys.argv)-1} files', flush=True)
h5files = [h5py.File(fname, 'r') for fname in fnames]
n_files = len(h5files)
t1 = time.monotonic()
print(f'Opening files done in {t1-t0:.2f}s.', flush=True)


t0 = time.monotonic()
print(f'Retreiving no. of images per file', flush=True)
n_images_per_file = [h5file['intensities'].shape[0] for h5file in h5files]
n_total_images = np.sum(n_images_per_file)
t1 = time.monotonic()
print(f'done in {t1-t0:.2f}s.', flush=True)


# Create a random no. identifier for each image in all datasets
t0 = time.monotonic()
print(f'Generating random indices', flush=True)
rand_ids = random.sample(range(n_total_images), n_total_images)
assert len(rand_ids) == np.unique(rand_ids).shape[0]
t1 = time.monotonic()
print(f'Create random indices for {n_total_images} done in {t1-t0:.2f}s.', flush=True)


# Combine intensities and orientations
print(f'Start copying input files', flush=True)
intensities = np.empty([n_total_images]+list(h5files[0]['intensities'].shape[1:]), dtype=h5files[0]['intensities'].dtype)
orientations = np.empty([n_total_images]+list(h5files[0]['orientations'].shape[1:]), dtype=h5files[0]['orientations'].dtype)
st = 0
for i, h5file in enumerate(h5files):
    t0 = time.monotonic()
    intensities[st:st+n_images_per_file[i]] = h5file['intensities'][:]
    orientations[st:st+n_images_per_file[i]] = h5file['orientations'][:]
    t1 = time.monotonic()
    print(f'copied st:{st} en:{st+n_images_per_file[i]} done in {t1-t0:.2f}s.', flush=True)
    st += n_images_per_file[i]


out_file = 'out.h5'
unique_keys = ['beam_offsets', 'fluences', 'pixel_index_map', 'pixel_position_reciprocal', 'polarization', 'solid_angle']
print(f'Start writing output file', flush=True)
t0 = time.monotonic()
with h5py.File(out_file, 'w') as hf:
    # These are the same for all conformations
    print(f'  Writing unique keys and their values', flush=True)
    for unique_key in unique_keys:
        hf.create_dataset(unique_key, data=h5files[0][unique_key][:])

    # Copy all volumes with new key and combine all command_lines
    print(f'  Collecting command lines', flush=True)
    command_line = ''
    for i, h5file in enumerate(h5files):
        #hf.create_dataset(f'volume_{Path(fnames[i]).stem}', data=h5file['volume'][:])
        command_line += h5file.attrs['command_line'] + ';'

    # Combine intensities and orientations in random orders by batches
    # to avoid memory problem
    print(f'  Writing intensities', flush=True)
    intensities_ds = hf.create_dataset('intensities', intensities.shape)
    st = 0
    for i in range(n_files):
        intensities_ds[st:st+n_images_per_file[i],:,:,:] = intensities[rand_ids[st:st+n_images_per_file[i]]]
        st += n_images_per_file[i]
        print(f'    done with batch {i}')


    print(f'  Writing orientations', flush=True)
    orientations_ds = hf.create_dataset('orientations', orientations.shape)
    st = 0
    for i in range(n_files):
        orientations_ds[st:st+n_images_per_file[i],:,:] = orientations[rand_ids[st:st+n_images_per_file[i]]]
        st += n_images_per_file[i]
        print(f'    done with batch {i}')

    hf.attrs['command_line'] = command_line
t1 = time.monotonic()
print(f'Writing {out_file} done in {t1-t0:.2f}s.', flush=True)

# Check the output files
with h5py.File(out_file, 'r') as hf:
    for key in hf.keys():
        print(f'{key} {hf[key].shape}')
        if key in unique_keys:
            assert np.array_equal(hf[key][:], h5files[0][key][:])

    for i in range(3):
        assert np.array_equal(hf['intensities'][i], intensities[rand_ids[i]])





