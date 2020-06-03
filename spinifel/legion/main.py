import h5py
import numpy as np
import os
import pygion
from pygion import acquire, attach_hdf5, task, Partition, Region, R, WD

from spinifel import parms


@task(privileges=[R])
def print_regions(pixel_index):
    print(pixel_index.map[0, 0, 0, 0])
    print(pixel_index.map[0, 0, 0, 1])


@task(privileges=[WD])
def load_pixel_index(pixel_index):
    f = h5py.File(parms.data_path, 'r')
    pixel_index.map[...] = f['pixel_index_map'][...]


@task(replicable=True)
def main():
    print("In Legion main", flush=True)

    N_images = 1

    det_shape = parms.det_shape
    data_type = getattr(pygion, parms.data_type_str)

    data = Region((N_images,) + det_shape, {'images': data_type})
    pixel_position = Region(det_shape + (3,), {'reciprocal': pygion.float32})
    pixel_index = Region(det_shape + (2,), {'map': pygion.int32})

    load_pixel_index(pixel_index)

    print_regions(pixel_index)
