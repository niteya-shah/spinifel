import numpy  as np
import PyNVTX as nvtx
import os
import pygion

from pygion import acquire, attach_hdf5, execution_fence, task, Partition, Region, R, Tunable, WD, RO

from spinifel import settings, utils, contexts, checkpoint
from spinifel.prep import save_mrc

from .prep import get_data, init_partitions_regions_psana2, load_image_batch, load_pixel_data, process_data
from .utils import union_partitions_with_stride, fill_region
from .autocorrelation import solve_ac
from .phasing import phase, prev_phase, cov, phased_output
from .orientation_matching import match, create_orientations_rp
from . import mapper
from . import checkpoint

@nvtx.annotate("legion/main.py", is_prefix=True)
def load_psana():
    logger = utils.Logger(True)
    assert settings.use_psana == True
    # Reading input images using psana2
    #assert settings.ps_smd_n_events == settings.N_images_per_rank
    # For now, we use one smd chunk per node just to keep things simple.
    # parameters used
    # settings.ps_batch_size -> minimum batch size
    # settings.
    assert settings.ps_smd_n_events == settings.N_images_per_rank
    from psana.psexp.tools import mode
    from psana import DataSource
    total_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    # minimum images to load per batch
    min_batch_size = settings.N_images_per_rank

    # for now assume N_images_max is a multiple of N_images_per_rank
    assert settings.N_images_max % min_batch_size == 0
    
    max_batches = settings.N_images_max//min_batch_size

    # preload a max of N_image_batches_max per iteration
    max_batches_per_iter = settings.N_image_batches_max

    # max_images_per_iter
    max_images_per_iter = settings.N_image_batches_max*min_batch_size
    
    logger.log(f'Using psana: exp={settings.ps_exp}, run={settings.ps_runnum}, dir={settings.ps_dir}, max_batches_per_iter={max_batches_per_iter}, max_batches={max_batches}, mode={mode}')
    assert mode == 'legion'
    ds = DataSource(exp=settings.ps_exp, run=settings.ps_runnum,
                    dir=settings.ps_dir)
    
    slices, all_partitions, slices_images, slices_images_p = init_partitions_regions_psana2()

    # load pixel_position, pixel_distance, pixel_index
    pixel_position, pixel_distance, pixel_index, run = load_pixel_data(ds)
    gen_run = ds.runs()
    gen_run, gen_smd, run = load_image_batch(run,gen_run,None,slices_images_p[0])

    pixel_position, pixel_distance, pixel_index = process_data(slices_images, slices_images_p[0], slices, all_partitions[0], pixel_distance, pixel_index, pixel_position,0)

    return pixel_position, pixel_distance, pixel_index, slices, all_partitions[0], all_partitions, gen_run, gen_smd, run, slices_images, slices_images_p


def load_psana_subset(gen_run, gen_smd, batch_size, cur_batch_size, slices, all_partitions, run,
                      slices_images, slices_images_all_p, idx, pixel_position, pixel_distance, pixel_index):

    # cur_batch_size  must be a multiple of batch_size
    assert cur_batch_size%batch_size == 0
    slices_p = all_partitions[cur_batch_size//batch_size]
    slices_images_p = slices_images_all_p[idx]
    gen_run, gen_smd, run = load_image_batch(run, gen_run, gen_smd, slices_images_p)

    # bin data
    pixel_position, pixel_distance, pixel_index = process_data(slices_images, slices_images_p, slices, slices_p, pixel_distance, pixel_index, pixel_position,idx)

    return slices_p, gen_run, gen_smd, run, pixel_distance, pixel_index, pixel_position


def main_spinifel(pixel_position, pixel_distance, pixel_index, slices_p, n_images_per_rank):
    logger = utils.Logger(True)
    timer = utils.Timer()
    curr_gen = 0

    # not supported for streaming/psana mode since image data grows
    assert(settings.load_gen == 0)

    orientations, orientations_p = create_orientations_rp(n_images_per_rank)
    solved = solve_ac(0, pixel_position, pixel_distance, slices_p)
    logger.log(f"AC recovered in {timer.lap():.2f}s.")

    phased = phase(0, solved)
    phased_output(phased, 0)

    #not valid since tasks are async
    #logger.log(f"Problem phased in {timer.lap():.2f}s.")

    # Use improvement of cc(prev_rho, cur_rho) to determine if we should
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
        match(
            phased, slices_p, pixel_position, pixel_distance, orientations_p, n_images_per_rank)
        logger.log(f"Orientations matched in {timer.lap():.2f}s.")

        # Solve autocorrelation
        solved = solve_ac(
            generation, pixel_position, pixel_distance, slices_p,
            orientations, orientations_p, phased)
        logger.log(f"AC recovered in {timer.lap():.2f}s.")

        prev_phased = prev_phase(generation, phased, prev_phased)

        phased = phase(generation, solved, phased)
        #logger.log(f"Problem phased in {timer.lap():.2f}s.")

        # Check if density converges
        if settings.chk_convergence:
            cov_xy, is_cov =  cov(prev_phased, phased, cov_xy, cov_delta)
        
            if is_cov:
                print("Stopping criteria met!")
                break;

        phased_output(phased, generation)

    #fill regions so they get garbage collected
    fill_region(orientations, 0.0)
    fill_region(phased, 0.0)
    fill_region(prev_phased, 0.0)

# read the data and run the main algorithm. This can be repeated
@nvtx.annotate("legion/main.py", is_prefix=True)
def main():
    logger = utils.Logger(True)
    logger.log("In Legion Psana2 with Streaming")
    ds = None
    timer = utils.Timer()

    # Reading input images using psana2
    assert(settings.use_psana)
    # compute pixel_position, pixel_distance, pixel_index, partitions,
    # regions
    (pixel_position, pixel_distance, pixel_index, slices, slices_p, all_partitions, gen_run, gen_smd, run, slices_images, slices_images_p) = load_psana()
    logger.log(f"Loaded in {timer.lap():.2f}s.")
    done = False
    batch_size = settings.N_images_per_rank
    cur_batch_size = batch_size
    max_batch_size = settings.N_images_max
    n_points = Tunable.select(Tunable.GLOBAL_PYS).get()
    while not done:
        # slices_p reflects union of new images that have been loaded
        # and earlier images in the previous partition
        if cur_batch_size != batch_size:
            slices_p = union_partitions_with_stride(slices,
                                                    settings.reduced_det_shape,
                                                    cur_batch_size,
                                                    max_batch_size,
                                                    max_batch_size,
                                                    n_points)
        main_spinifel(pixel_position, pixel_distance, pixel_index, slices_p, cur_batch_size)

        if cur_batch_size == max_batch_size:
            done = True
        if not done:
            for i in range(settings.N_image_batches_max):
                slices_p, gen_run, gen_smd, run, pixel_distance, pixel_index, pixel_position = load_psana_subset(gen_run, gen_smd, batch_size, cur_batch_size, slices, all_partitions, run, slices_images, slices_images_p, i, pixel_position, pixel_distance, pixel_index)
                cur_batch_size = cur_batch_size + batch_size
                if settings.verbose:
                    print(f'cur_batch_size = {cur_batch_size}, batch_size = {batch_size}, N_image_batches_max={settings.N_image_batches_max}, max_batch_size={max_batch_size}')
                if cur_batch_size==max_batch_size:
                    break

    execution_fence(block=True)
    logger.log(f"Results saved in {settings.out_dir}")
    logger.log(f"Successfully completed in {timer.total():.2f}s.")

            
