import h5py
import numpy
import os
import pygion
from pygion import acquire, attach_hdf5, task, Region, R

from spinifel import parms


@task(privileges=[R])
def print_region(data):
    print(data.images)


@task(replicable=True)
def main():
    print("In Legion main", flush=True)

    data = Region((parms.N_images,) + parms.det_shape,
                  {'images': pygion.float64})

    with attach_hdf5(data, str(parms.data_path), {'images': 'slices'},
                     pygion.file_read_only):
        pass


if __name__ == '__main__':
    main()
