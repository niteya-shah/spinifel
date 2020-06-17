from spinifel import parms, utils

from .prep import get_data
from .autocorrelation import solve_ac
from .phasing import phase
from .orientation_matching import match


def main():
    print("In sequential main", flush=True)

    N_images = parms.N_images_per_rank
    det_shape = parms.det_shape

    timer = utils.Timer()

    (pixel_position_reciprocal,
     pixel_distance_reciprocal,
     pixel_index_map,
     slices_) = get_data(N_images)
    print(f"Loaded in {timer.lap():.2f}s.")

    ac, it_count = solve_ac(
        0, pixel_position_reciprocal, pixel_distance_reciprocal, slices_)
    print(f"AC recovered in {timer.lap():.2f}s.")

    ac_phased, support_, rho_ = phase(0, ac)
    print(f"Problem phased in {timer.lap():.2f}s.")

    orientations = match(
        ac_phased, slices_,
        pixel_position_reciprocal, pixel_distance_reciprocal)
    print(f"Orientations matched in {timer.lap():.2f}s.")

    ac, it_count = solve_ac(
        1, pixel_position_reciprocal, pixel_distance_reciprocal, slices_,
        orientations, ac_phased)
    print(f"AC recovered in {timer.lap():.2f}s.")

    ac_phased, support_, rho_ = phase(1, ac, support_, rho_)
    print(f"Problem phased in {timer.lap():.2f}s.")

    print(f"Total: {timer.total():.2f}s.")
