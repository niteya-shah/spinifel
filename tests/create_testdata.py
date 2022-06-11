import h5py

test_cases = ['3iyf', '2cex']
in_files = ['/global/cfs/cdirs/m2859/data/3iyf/clean/3iyf_sim_400k.h5',
       '/global/cfs/cdirs/m2859/data/2cex/2cexa/clean/2cex_128x128pixels_500k.h5']
out_files = ['/global/cfs/cdirs/m2859/data/testdata/3IYF/3iyf_sim_10k.h5', 
       '/global/cfs/cdirs/m2859/data/testdata/2CEX/2cex_sim_10k.h5']
N_sample_images = 10000
det_X  = 128
det_Y  = 128

for test_case, in_file, out_file in zip(test_cases, in_files, out_files):
    print(f'Test case: {test_case}')

    # Check for expected data
    expected_keys = ['orientations',
                     'intensities',
                     'pixel_position_reciprocal',
                     'pixel_index_map',
                     'volume',]
    fi = h5py.File(in_file, 'r')
    for key in expected_keys:
        print(f'key:{key} shape:{fi[key].shape}')
    
    # Setup expected shapes
    expected_shapes = [(N_sample_images, 1, 4),
            (N_sample_images, 1, det_X, det_Y),
            (1, det_X, det_Y, 3),
            (1, det_X, det_Y, 2),
            fi['volume'].shape]

    print(f'writing out {N_sample_images} to {out_file}')
    with h5py.File(out_file, 'w') as fo:
        for ekey, eshape  in zip(expected_keys, expected_shapes):
            in_data = fi[ekey]
            dset = fo.create_dataset(ekey, eshape, dtype=in_data.dtype) 
            if ekey in ('orientations', 'intensities'):
                dset[:] = in_data[:N_sample_images][:]
            else:
                dset[:] = in_data

    print(f'output file contents:')
    fo = h5py.File(out_file, 'r')
    for key in expected_keys:
        print(f'key:{key} shape:{fo[key].shape}')
    
    print('Done\n')



