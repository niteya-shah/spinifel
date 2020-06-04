import h5py
import numpy as np
import os
import pygion
from pygion import acquire, attach_hdf5, task, Partition, Region, R, WD

from spinifel import parms


@task(privileges=[WD])
def load_pixel_position(pixel_position):
    pixel_position_reciprocal = pixel_position.reciprocal
    with h5py.File(parms.data_path, 'r') as h5f:
        pixel_position_reciprocal[:] = np.moveaxis(
            h5f['pixel_position_reciprocal'][:], -1, 0)


@task(privileges=[WD])
def load_pixel_index(pixel_index):
    pixel_index_map = pixel_index.map
    with h5py.File(parms.data_path, 'r') as h5f:
        pixel_index_map[:] = np.moveaxis(
            h5f['pixel_index_map'][:], -1, 0)


def get_data():
    pixel_position = Region((3,) + parms.det_shape, {'reciprocal': pygion.float32})
    pixel_index = Region((2,) + parms.det_shape, {'map': pygion.int32})

    load_pixel_position(pixel_position)
    load_pixel_index(pixel_index)

    return (pixel_position, pixel_index)
