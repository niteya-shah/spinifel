import numpy as np
import os
import pygion
from pygion import acquire, attach_hdf5, task, Partition, Region, R, WD

from spinifel import parms

from .prep import get_data
from .autocorrelation import solve_ac
from .phasing import phase
from .orientation_matching import match


@task(replicable=True)
def main():
    print("In Legion main", flush=True)

    (pixel_position,
     pixel_distance,
     pixel_index,
     slices, slices_p) = get_data()

    solved = solve_ac(0, pixel_position, pixel_distance, slices, slices_p)

    phased = phase(0, solved)

    for generation in range(1, 10):
        orientations, orientations_p = match(
            phased, slices, slices_p, pixel_position, pixel_distance)

        solved = solve_ac(
            generation, pixel_position, pixel_distance, slices, slices_p,
            orientations, orientations_p, phased)

        phased = phase(generation, solved, phased)
