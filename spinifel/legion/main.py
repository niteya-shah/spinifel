import numpy  as np
import PyNVTX as nvtx
import os
import pygion

from pygion import acquire, attach_hdf5, execution_fence, task, Partition, Region, R, Tunable, WD

from spinifel import parms, utils, SpinifelSettings
from spinifel.prep import save_mrc

from .prep import get_data
from .autocorrelation import setup_solve_ac, solve_ac
from .phasing import phase, prev_phase, cov
from .orientation_matching import match
from . import mapper


@task(replicable=True)
@nvtx.annotate("legion/main.py", is_prefix=True)
def main():

    timer = utils.Timer()

    total_procs = Tunable.select(Tunable.GLOBAL_PYS).get()

    # Reading input images from hdf5
    N_images_per_rank = parms.N_images_per_rank
    batch_size = min(N_images_per_rank, 100)
    max_events = min(parms.N_images_max, total_procs*N_images_per_rank)
    writer_rank = 0 # pick writer rank as core 0

    ds = None
    if parms.use_psana:
        # For now, we use one smd chunk per node just to keep things simple.
        # os.environ['PS_SMD_N_EVENTS'] = str(N_images_per_rank)
        settings                 = SpinifelSettings()
        settings.ps_smd_n_events = N_images_per_rank

        from psana import DataSource
        logger.log("Using psana")
        ds = DataSource(exp=parms.exp, run=parms.runnum, dir=parms.data_dir,
                        batch_size=batch_size, max_events=max_events)

    # Setup logger after knowing the writer rank 
    logger = utils.Logger(True)
    logger.log("In Legion main") 

    # Load unique set of intensity slices for python process
    (pixel_position,
     pixel_distance,
     pixel_index,
     slices, slices_p,
     orientations_prior, orientations_prior_p) = get_data(ds)
    logger.log(f"Loaded in {timer.lap():.2f}s.")

    # Generation 0: solve_ac and phase 
    N_generations = parms.N_generations
    generation = 0
    logger.log(f"#"*27)
    logger.log(f"##### Generation {generation}/{N_generations} #####")
    logger.log(f"#"*27)


    (orientations, orientations_p,
     nonuniform, nonuniform_p,
     nonuniform_v, nonuniform_v_p,
     ac, uregion, uregion_ups,
     results, results_p,
     summary, summary_p) = setup_solve_ac(pixel_position, pixel_distance)

    solved = solve_ac(0, pixel_position, pixel_distance,
                      slices, slices_p,
                      orientations_p, nonuniform_p, nonuniform_v_p,
                      ac, uregion, uregion_ups,
                      results_p, summary, summary_p)
    logger.log(f"AC recovered in {timer.lap():.2f}s.")

    phased = phase(0, solved)
    logger.log(f"Problem phased in {timer.lap():.2f}s.")

    # Use improvement of cc(prev_rho, cur_rho) to dertemine if we should
    # terminate the loop
    prev_phased = None
    cov_xy = 0
    cov_delta = .05
    
    for generation in range(1, N_generations):
        logger.log(f"#"*27)
        logger.log(f"##### Generation {generation}/{N_generations} #####")
        logger.log(f"#"*27)

        # Orientation matching
        orientations, orientations_p =  match(phased, slices, slices_p, pixel_position, pixel_distance, orientations, orientations_p)
        logger.log(f"Orientations matched in {timer.lap():.2f}s.")

        # Solve autocorrelation
        solved = solve_ac(generation, pixel_position, pixel_distance,
                          slices, slices_p,
                          orientations_p, nonuniform_p, nonuniform_v_p,
                          ac, uregion, uregion_ups,
                          results_p, summary, summary_p, phased)
        logger.log(f"AC recovered in {timer.lap():.2f}s.")

        #prev_phased = prev_phase(generation, phased, prev_phased)

        phased = phase(generation, solved, phased)
        logger.log(f"Problem phased in {timer.lap():.2f}s.")
        
        # Check if density converges
        if parms.chk_convergence:
            cov_xy, is_cov =  cov(prev_phased, phased, cov_xy, cov_delta)
        
            if is_cov:
                print("Stopping criteria met!")
                break;

        rho = np.fft.ifftshift(phased.rho_)

        save_mrc(parms.out_dir / f"ac-{generation}.mrc", phased.ac)
        save_mrc(parms.out_dir / f"rho-{generation}.mrc", rho)

    execution_fence(block=True)

    logger.log(f"Results saved in {parms.out_dir}")
    logger.log(f"Successfully completed in {timer.total():.2f}s.")
