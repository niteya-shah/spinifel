from spinifel import parms
from spinifel.sequential import get_data, solve_ac


def main():
    print("In sequential main", flush=True)

    N_images = 1000
    det_shape = parms.det_shape

    (pixel_position_reciprocal,
     pixel_distance_reciprocal,
     pixel_index_map,
     slices_) = get_data(N_images)

    solve_ac(N_images,
             pixel_position_reciprocal,
             pixel_distance_reciprocal,
             slices_)
