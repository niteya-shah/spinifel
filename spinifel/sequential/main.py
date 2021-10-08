import PyNVTX as nvtx

from spinifel import settings, utils

from .prep import get_data
from .autocorrelation import solve_ac
from .phasing import phase
from .orientation_matching import slicing_and_match



@nvtx.annotate("sequential/main.py", is_prefix=True)
def main():
    logger = utils.Logger(True)
    logger.log("In sequential main")

    N_images = settings.N_images_per_rank
    det_shape = settings.det_shape

    timer = utils.Timer()

    ds = None
    if settings.use_psana:
        from psana import DataSource
        logger.log("Using psana")
        ds = DataSource(exp=settings.exp, run=settings.runnum,
                        dir=settings.data_dir, batch_size=50,
                        max_events=settings.N_images_max)

    (pixel_position_reciprocal,
     pixel_distance_reciprocal,
     pixel_index_map,
     slices_) = get_data(N_images, ds)
    logger.log(f"Loaded in {timer.lap():.2f}s.")

    ac = solve_ac(
        0, pixel_position_reciprocal, pixel_distance_reciprocal, slices_)
    logger.log(f"AC recovered in {timer.lap():.2f}s.")

    ac_phased, support_, rho_ = phase(0, ac)
    logger.log(f"Problem phased in {timer.lap():.2f}s.")

    for generation in range(1, 10):
        orientations = slicing_and_match(
            ac_phased, slices_,
            pixel_position_reciprocal, pixel_distance_reciprocal)
        logger.log(f"Orientations matched in {timer.lap():.2f}s.")

        ac = solve_ac(
            generation, pixel_position_reciprocal, pixel_distance_reciprocal,
            slices_, orientations, ac_phased)
        logger.log(f"AC recovered in {timer.lap():.2f}s.")

        ac_phased, support_, rho_ = phase(generation, ac, support_, rho_)
        logger.log(f"Problem phased in {timer.lap():.2f}s.")

    logger.log(f"Total: {timer.total():.2f}s.")
