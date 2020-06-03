from mpi4py import MPI

from spinifel import parms, utils

from .prep import get_data


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    if rank == 0:
        print("In MPI main", flush=True)

    N_images_per_rank = parms.N_images_per_rank

    if rank == 0:
        timer = utils.Timer()

    (pixel_position_reciprocal,
     pixel_distance_reciprocal,
     pixel_index_map,
     slices_) = get_data(N_images_per_rank)

    if rank == 0:
        print(f"Loaded in {timer.lap():.2f}s.")
        print(f"Total: {timer.total():.2f}s.")
