import numpy as np
import os
import pygion
from pygion import acquire, attach_hdf5, task, Partition, Region, R, Tunable, WD

from spinifel import parms

from .prep import get_data
from .autocorrelation import solve_ac
from .phasing import phase
from .orientation_matching import match


@task(replicable=True)
def main():
    print("In Legion main", flush=True)

    total_procs = Tunable.select(Tunable.GLOBAL_PYS).get()

    N_images_per_rank = parms.N_images_per_rank
    batch_size = min(N_images_per_rank, 100)
    max_events = min(parms.N_images_max, total_procs*N_images_per_rank)

    ds = None
    if parms.use_psana:
        # For now, we use one smd chunk per node just to keep things simple.
        os.environ['PS_SMD_N_EVENTS'] = str(N_images_per_rank)

        from psana import DataSource
        ds = DataSource(exp=parms.exp, run=parms.runnum, dir=parms.data_dir,
                        batch_size=batch_size, max_events=max_events)

    (pixel_position,
     pixel_distance,
     pixel_index,
     slices, slices_p) = get_data(ds)

    solved = solve_ac(0, pixel_position, pixel_distance, slices, slices_p)

    phased = phase(0, solved)

    for generation in range(1, 10):
        orientations, orientations_p = match(
            phased, slices, slices_p, pixel_position, pixel_distance)

        solved = solve_ac(
            generation, pixel_position, pixel_distance, slices, slices_p,
            orientations, orientations_p, phased)

        phased = phase(generation, solved, phased)
