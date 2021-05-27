from mpi4py import MPI

from spinifel import parms, utils
from spinifel.prep import save_mrc

from .prep import get_data
from .autocorrelation import solve_ac
from .phasing import phase
from .orientation_matching import match

import numpy as np

import PyNVTX as nvtx


@nvtx.annotate("mpi/main.py", is_prefix=True)
def main():
    comm = MPI.COMM_WORLD

    logger = utils.Logger(comm.rank==(2 if parms.use_psana else 0))
    logger.log("In MPI main")

    N_images_per_rank = parms.N_images_per_rank

    timer = utils.Timer()

    ds = None
    if parms.use_psana:
        from psana import DataSource
        logger.log("Using psana")
        N_big_data_nodes = comm.size - 2
        batch_size = min(N_images_per_rank, 100)
        max_events = min(parms.N_images_max, N_big_data_nodes*N_images_per_rank)
        def destination(timestamp):
            # Return big data node destination, numbered from 1, round-robin
            destination.last = destination.last % N_big_data_nodes + 1
            return destination.last
        destination.last = 0
        ds = DataSource(exp=parms.exp, run=parms.runnum, dir=parms.data_dir,
                        destination=destination, max_events=max_events)

    (pixel_position_reciprocal,
     pixel_distance_reciprocal,
     pixel_index_map,
     slices_) = get_data(N_images_per_rank, ds)
    logger.log(f"Loaded in {timer.lap():.2f}s.")

    ac = solve_ac(
        0, pixel_position_reciprocal, pixel_distance_reciprocal, slices_)
    logger.log(f"AC recovered in {timer.lap():.2f}s.")

    ac_phased, support_, rho_ = phase(0, ac)
    logger.log(f"Problem phased in {timer.lap():.2f}s.")

    # Use improvement of cc(prev_rho, cur_rho) to dertemine if
    # we should terminate the loop
    cov_xy = 0
    cov_delta = .05

    N_generations = parms.N_generations
    for generation in range(1, N_generations):
        print('generation =', generation)
        orientations = match(
            ac_phased, slices_,
            pixel_position_reciprocal, pixel_distance_reciprocal)
        logger.log(f"Orientations matched in {timer.lap():.2f}s.")

        ac = solve_ac(
            generation, pixel_position_reciprocal, pixel_distance_reciprocal,
            slices_, orientations, ac_phased)
        logger.log(f"AC recovered in {timer.lap():.2f}s.")

        if comm.rank == 0: prev_rho_ = rho_[:]
        ac_phased, support_, rho_ = phase(generation, ac, support_, rho_)

        if comm.rank == 0:
            cc_matrix = np.corrcoef(prev_rho_.flatten(), rho_.flatten())
            prev_cov_xy = cov_xy
            cov_xy = cc_matrix[0,1]
        else:
            prev_cov_xy = None
            cov_xy = None

        logger.log(f"Problem phased in {timer.lap():.2f}s. cc={cov_xy:.2f} delta={cov_xy-prev_cov_xy:.2f}")
        if cov_xy - prev_cov_xy < cov_delta:
            break

        rho = np.fft.ifftshift(rho_)
        print("rho =", rho)

        if comm.rank == 0:
            save_mrc(parms.out_dir / f"ac-{generation}.mrc", ac_phased)
            save_mrc(parms.out_dir / f"rho-{generation}.mrc", rho)
            np.save(parms.out_dir / f"ac-{generation}.npy", ac_phased)
            np.save(parms.out_dir / f"rho-{generation}.npy", rho)

    logger.log(f"Total: {timer.total():.2f}s.")

