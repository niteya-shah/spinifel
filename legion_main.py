import h5py
import numpy as np
import os
import pygion
from pygion import acquire, attach_hdf5, task, Partition, Region, R

from spinifel import parms


@task(privileges=[R])
def print_region(data):
    print(data.images)


@task(replicable=True)
def main():
    print("In Legion main", flush=True)

    N_images = 1

    det_shape = parms.det_shape
    data_type = getattr(pygion, parms.data_type_str)

    data = Region((N_images,) + det_shape, {'images': data_type})

    with attach_hdf5(data, str(parms.data_path),
                     {'images': parms.data_field_name},
                     pygion.file_read_only):
        print_region(data)


if __name__ == '__main__':
    main()
