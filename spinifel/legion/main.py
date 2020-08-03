import numpy as np
import os
import pygion
from pygion import acquire, attach_hdf5, task, Partition, Region, R, WD

from spinifel import parms

from .prep import get_data
from .autocorrelation import solve_ac
from .phasing import phase


@task(replicable=True)
def main():
    print("In Legion main", flush=True)

    (pixel_position,
     pixel_distance,
     pixel_index,
     slices, slices_p) = get_data()

    solved = solve_ac(0, pixel_position, pixel_distance, slices, slices_p)

    phased = phase(0, solved)
