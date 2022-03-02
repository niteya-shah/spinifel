import numpy  as np
import PyNVTX as nvtx
import os
import pygion

from pygion import acquire, attach_hdf5, execution_fence, task, Partition, Region, R, Tunable, WD

from spinifel import settings, utils, contexts, checkpoint
from spinifel.prep import save_mrc

from .prep import get_data
from .autocorrelation import solve_ac
from .phasing import phase, prev_phase, cov
from .orientation_matching import match
from . import mapper
from . import checkpoint

@task(replicable=True)
@nvtx.annotate("legion/main.py", is_prefix=True)
def main():

    timer = utils.Timer()

    total_procs = Tunable.select(Tunable.GLOBAL_PYS).get()

    # Reading input images from hdf5
    N_images_per_rank = settings.N_images_per_rank
    batch_size = min(N_images_per_rank, 100)
    max_events = min(settings.N_images_max, total_procs*N_images_per_rank)
    writer_rank = 0 # pick writer rank as core 0

    # Reading input images using psana2
    ds = None
    if settings.use_psana:
        # For now, we use one smd chunk per node just to keep things simple.
        # os.environ['PS_SMD_N_EVENTS'] = str(N_images_per_rank)
        settings.ps_smd_n_events = N_images_per_rank

        from psana import DataSource
        logger.log("Using psana")
        ds = DataSource(exp=settings.exp, run=settings.runnum,
                        dir=settings.data_dir, batch_size=batch_size,
                        max_events=max_events)

    # Setup logger after knowing the writer rank
    logger = utils.Logger(True)
    logger.log("In Legion main")

    # Load unique set of intensity slices for python process
    (pixel_position,
     pixel_distance,
     pixel_index,
     slices, slices_p) = get_data(ds)
    logger.log(f"Loaded in {timer.lap():.2f}s.")

    if settings.load_gen > 0: # Load input from previous generation
        curr_gen = settings.load_gen
        phased, orientations, orientations_p = checkpoint.load_checkpoint(settings.out_dir, settings.load_gen)
    else:
        solved = solve_ac(0, pixel_position, pixel_distance, slices, slices_p)
        logger.log(f"AC recovered in {timer.lap():.2f}s.")

        phased = phase(0, solved)
        logger.log(f"Problem phased in {timer.lap():.2f}s.")

        rho = np.fft.ifftshift(phased.rho_)
        print('rho =', rho)

        save_mrc(settings.out_dir / f"ac-0.mrc", phased.ac)
        save_mrc(settings.out_dir / f"rho-0.mrc", rho)

    # Use improvement of cc(prev_rho, cur_rho) to dertemine if we should
    # terminate the loop
    prev_phased = None
    cov_xy = 0
    cov_delta = .05
    curr_gen +=1

    N_generations = settings.N_generations
    for generation in range(curr_gen, N_generations+1):
        logger.log(f"#"*27)
        logger.log(f"##### Generation {generation}/{N_generations} #####")
        logger.log(f"#"*27)

        # Orientation matching
        orientations, orientations_p = match(
            phased, slices, slices_p, pixel_position, pixel_distance)
        logger.log(f"Orientations matched in {timer.lap():.2f}s.")

        # Solve autocorrelation
        solved = solve_ac(
            generation, pixel_position, pixel_distance, slices, slices_p,
            orientations, orientations_p, phased)
        logger.log(f"AC recovered in {timer.lap():.2f}s.")

        prev_phased = prev_phase(generation, phased, prev_phased)

        phased = phase(generation, solved, phased)
        logger.log(f"Problem phased in {timer.lap():.2f}s.")

        # Check if density converges
        if settings.chk_convergence:
            cov_xy, is_cov =  cov(prev_phased, phased, cov_xy, cov_delta)
        
            if is_cov:
                print("Stopping criteria met!")
                break;

        rho = np.fft.ifftshift(phased.rho_)
        print('rho =', rho)

        save_mrc(settings.out_dir / f"ac-{generation}.mrc", phased.ac)
        save_mrc(settings.out_dir / f"rho-{generation}.mrc", rho)

    execution_fence(block=True)

    logger.log(f"Results saved in {settings.out_dir}")
    logger.log(f"Successfully completed in {timer.total():.2f}s.")
        

