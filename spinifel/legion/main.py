import numpy  as np
import PyNVTX as nvtx
import os
import pygion

from pygion import acquire, attach_hdf5, task, Partition, Region, R, Tunable, WD

from spinifel import settings

from .prep import get_data
from .autocorrelation import solve_ac
from .phasing import phase, prev_phase, cov
from .orientation_matching import match
from . import mapper


@task(replicable=True)
@nvtx.annotate("legion/main.py", is_prefix=True)
def main():
    print("In Legion main", flush=True)

    total_procs = Tunable.select(Tunable.GLOBAL_PYS).get()

    N_images_per_rank = settings.N_images_per_rank
    batch_size = min(N_images_per_rank, 100)
    max_events = min(settings.N_images_max, total_procs*N_images_per_rank)

    ds = None
    if settings.use_psana:
        # For now, we use one smd chunk per node just to keep things simple.
        # os.environ['PS_SMD_N_EVENTS'] = str(N_images_per_rank)
        settings.ps_smd_n_events = N_images_per_rank

        from psana import DataSource
        ds = DataSource(exp=settings.exp, run=settings.runnum,
                        dir=settings.data_dir, batch_size=batch_size,
                        max_events=max_events)

    (pixel_position,
     pixel_distance,
     pixel_index,
     slices, slices_p) = get_data(ds)

    solved = solve_ac(0, pixel_position, pixel_distance, slices, slices_p)

    phased = phase(0, solved)
    prev_phased = None
    cov_xy = 0
    cov_delta = .05
    N_generations = settings.N_generations
    for generation in range(1, N_generations):
        orientations, orientations_p = match(
            phased, slices, slices_p, pixel_position, pixel_distance)

        solved = solve_ac(
            generation, pixel_position, pixel_distance, slices, slices_p,
            orientations, orientations_p, phased)

        prev_phased = prev_phase(generation, phased, prev_phased)

        phased = phase(generation, solved, phased)

        cov_xy, is_cov =  cov(prev_phased, phased, cov_xy, cov_delta)
        
        if is_cov:
            break;
        

