import h5py
import numpy as np
import os
import pygion
from pathlib import Path
from pygion import acquire, attach_hdf5, task, Partition, Region, R


@task(privileges=[R])
def print_region(R):
    print(R.x)


@task(replicable=True)
def main():
    print("In main", flush=True)

    R = Region([4, 4],
               {'x': pygion.int32, 'y': pygion.int32,
                'z': pygion.int32, 'w': pygion.int32})

    filename = str(Path(os.environ.get("DATA_DIR", "")) / "test.h5")

    with attach_hdf5(R, filename,
                     {'x': 'x', 'y': 'y', 'z': 'z', 'w': 'w'},
                     pygion.file_read_only):
        with acquire(R, ['x', 'y']):
            print_region(R)


if __name__ == '__main__':
    main()
