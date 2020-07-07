import numpy as np
import os
import pygion
from pygion import acquire, attach_hdf5, task, Partition, Region, R, WD

from spinifel import parms

from .prep import get_data
from .autocorrelation import solve_ac


@task(privileges=[R])
def print_region(region):
    for field in region.keys():
        value = getattr(region, field).flatten()[0]
        print(f"{field}: {value}")


@task(replicable=True)
def main():
    print("In Legion main", flush=True)

    (pixel_position,
     pixel_distance,
     pixel_index,
     slices, slices_p) = get_data()

    solve_ac()
