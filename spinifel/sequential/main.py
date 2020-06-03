from spinifel import parms, utils

from .prep import get_data
from .autocorrelation import solve_ac


def main():
    print("In sequential main", flush=True)

    N_images = parms.N_images
    det_shape = parms.det_shape

    timer = utils.Timer()

    (pixel_position_reciprocal,
     pixel_distance_reciprocal,
     pixel_index_map,
     slices_) = get_data(N_images)

    print(f"Loaded in {timer.lap():.2f}s.")

    ac, it_count = solve_ac(
        pixel_position_reciprocal, pixel_distance_reciprocal, slices_)

    print(f"AC recovered in {timer.lap():.2f}s.")

    print(f"Total: {timer.total():.2f}s.")
