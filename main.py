import h5py
import numpy
import os
import pygion
from pathlib import Path
from pygion import acquire, attach_hdf5, task, Fspace, Ispace, Region, R

@task(privileges=[R])
def print_region(data):
    print(data.images)

@task
def main():
    print("In main", flush=True)

    det_shape = (4, 512, 512)
    N_images = 1000

    data_dir = Path(os.environ.get("DATA_DIR", ""))
    data_path = data_dir / "2CEX-1.h5"

    data = Region((N_images,) + det_shape, {'images': pygion.float64})

    with attach_hdf5(data, str(data_path), {'images': 'slices'}, pygion.file_read_only):
        pass

if __name__ == '__main__':
    main()
