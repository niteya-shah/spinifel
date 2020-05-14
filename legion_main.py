import h5py
import numpy as np
import os
import pygion
from pygion import acquire, attach_hdf5, task, Partition, Region, R

from spinifel import parms


@task(privileges=[R, R, R])
def print_regions(data, pixel_position, pixel_index):
    print(data.images.flatten()[0])
    print(pixel_position.reciprocal.flatten()[0])
    print(pixel_index.map.flatten()[0])


@task(replicable=True)
def main():
    print("In Legion main", flush=True)

    N_images = 1

    det_shape = parms.det_shape
    data_type = getattr(pygion, parms.data_type_str)

    data = Region((N_images,) + det_shape, {'images': data_type})
    pixel_position = Region(det_shape + (3,), {'reciprocal': pygion.float32})
    pixel_index = Region(det_shape + (2,), {'map': pygion.int32})

    with attach_hdf5(data, str(parms.data_path),
                     {'images': parms.data_field_name},
                     pygion.file_read_only), \
         attach_hdf5(pixel_position, str(parms.data_path),
                     {'reciprocal': 'pixel_position_reciprocal'},
                     pygion.file_read_only), \
         attach_hdf5(pixel_index, str(parms.data_path),
                     {'map': 'pixel_index_map'},
                     pygion.file_read_only):
        print_regions(data, pixel_position, pixel_index)


if __name__ == '__main__':
    main()
