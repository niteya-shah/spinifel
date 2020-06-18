from mpi4py import MPI

from spinifel import parms, utils

from .prep import get_data
from .autocorrelation import solve_ac
from .phasing import phase
from .orientation_matching import match


def main():
    comm = MPI.COMM_WORLD

    logger = utils.Logger(comm.rank==0)
    logger.log("In MPI main")

    N_images_per_rank = parms.N_images_per_rank

    timer = utils.Timer()

    (pixel_position_reciprocal,
     pixel_distance_reciprocal,
     pixel_index_map,
     slices_) = get_data(N_images_per_rank)
    logger.log(f"Loaded in {timer.lap():.2f}s.")

    ac, it_count = solve_ac(
        0, pixel_position_reciprocal, pixel_distance_reciprocal, slices_)
    logger.log(f"AC recovered in {timer.lap():.2f}s.")

    ac_phased, support_, rho_ = phase(0, ac)
    logger.log(f"Problem phased in {timer.lap():.2f}s.")

    orientations = match(
        ac_phased, slices_,
        pixel_position_reciprocal, pixel_distance_reciprocal)
    logger.log(f"Orientations matched in {timer.lap():.2f}s.")

    ac, it_count = solve_ac(
        1, pixel_position_reciprocal, pixel_distance_reciprocal,
        slices_, orientations, ac_phased)
    logger.log(f"AC recovered in {timer.lap():.2f}s.")

    ac_phased, support_, rho_ = phase(1, ac, support_, rho_)
    logger.log(f"Problem phased in {timer.lap():.2f}s.")

    logger.log(f"Total: {timer.total():.2f}s.")
