import numpy as np
import PyNVTX as nvtx
import os
import pygion

from pygion import (
    acquire,
    attach_hdf5,
    execution_fence,
    task,
    Partition,
    Region,
    R,
    Tunable,
    WD,
    RO,
)

from spinifel import settings, utils, contexts, checkpoint
from spinifel.prep import save_mrc

from .prep import get_data, prep_objects_multiple
from .autocorrelation import solve_ac, solve_ac_conf
from .phasing import new_phase, create_phased_regions, phased_output, new_phase_conf, phased_output_conf
from .orientation_matching import match, create_orientations_rp, match_conf, create_min_dist_rp
from . import mapper
from . import checkpoint
from . import utils as lgutils
from .fsc import init_fsc_task, compute_fsc_task, check_convergence_task, compute_fsc_conf, check_convergence_conf


@nvtx.annotate("legion/main.py", is_prefix=True)
def load_psana():
    logger = utils.Logger(True, settings)
    assert settings.use_psana == True
    # Reading input images using psana2
    # For now, we use one smd chunk per node just to keep things simple.
    assert settings.ps_smd_n_events == settings.N_images_per_rank
    from psana.psexp.tools import mode
    from psana import DataSource

    total_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    batch_size = min(settings.N_images_per_rank, 100)
    max_events = total_procs * settings.N_images_per_rank
    logger.log(
        f"Using psana: exp={settings.ps_exp}, run={settings.ps_runnum}, dir={settings.ps_dir}, batch_size={batch_size}, max_events={max_events}, mode={mode}"
    )
    assert mode == "legion"
    ds = DataSource(
        exp=settings.ps_exp,
        run=settings.ps_runnum,
        dir=settings.ps_dir,
        max_events=max_events,
    )

    # Load unique set of intensity slices for python process
    (pixel_position, pixel_distance, pixel_index, slices, slices_p) = get_data(ds)
    return pixel_position, pixel_distance, pixel_index, slices, slices_p

@task(inner=True, privileges=[RO, RO, RO, RO])
@lgutils.gpu_task_wrapper
def main_task_conf(pixel_position, pixel_distance, pixel_index, slices, slices_p):
    logger = utils.Logger(True, settings)
    timer = utils.Timer()
    curr_gen = 0
    fsc = []
    total_procs = Tunable.select(Tunable.GLOBAL_PYS).get()

    ready_objs, ready_objs_p = lgutils.create_distributed_region(
        Tunable.select(Tunable.GLOBAL_PYS).get(), {"done": pygion.bool_}, ()
    )

    orientations = []
    orientations_p = []
    # create orientation regions for each conformation
    for i in range(settings.N_conformations):
        orientation_region, orientation_part = create_orientations_rp(
            settings.N_images_per_rank
        )
        orientations.append(orientation_region)
        orientations_p.append(orientation_part)

    # create min_dist -> N_procs x N_images_per_rank x N_conformations
    # create partitions
    # min_dist_p -> N_procs x N_conformations [N_images_per_rank]
    # min_dist_proc -> N_procs [N_images_per_rank x N_conformations]
    conf_regions_dict = create_min_dist_rp(
        settings.N_images_per_rank, settings.N_conformations
    )
    min_dist =  conf_regions_dict["min_dist"]
    min_dist_p = conf_regions_dict["min_dist_p"]
    min_dist_proc = conf_regions_dict["min_dist_proc"]
    conf = conf_regions_dict["conf"]
    conf_p = conf_regions_dict["conf_p"]

    # all partitions/regions need to be ready/available
    execution_fence(block=True)

    prep_objects_multiple(pixel_position, pixel_distance, slices_p, ready_objs_p, total_procs)

    if settings.pdb_path.is_file() and settings.chk_convergence:
        for i in range (settings.N_conformations):
            # an array of futures
            fsc_future_entry = init_fsc_task(pixel_distance,point=0)
            fsc.append(fsc_future_entry)

    solved, solve_ac_dict = solve_ac_conf(
        None, 0, pixel_position, pixel_distance, slices_p, ready_objs_p, conf_p, fsc)
    phased, phased_regions_dict = new_phase_conf(0, solved, fsc)
    phased_output_conf(phased, 0)
    curr_gen += 1

    N_generations = settings.N_generations
    for generation in range(curr_gen, N_generations + 1):
        logger.log(f"#" * 40)
        logger.log(
            f"##### Generation {generation}/{N_generations}:  #####"
        )
        logger.log(f"#" * 40)

        # Orientation matching
        match_conf(phased, orientations_p, slices_p, min_dist_p, min_dist_proc, conf_p, settings.N_images_per_rank, ready_objs_p, fsc)

        # Solve autocorrelation
        solved, solve_ac_dict = solve_ac_conf(
            solve_ac_dict,
            generation,
            pixel_position,
            pixel_distance,
            slices_p,
            ready_objs_p,
            conf_p,
            fsc,
            orientations,
            orientations_p,
            phased
        )

        phased, phased_regions_dict = new_phase_conf(
            generation, solved, fsc, phased_regions_dict
        )
        # async tasks logger.log(f"Problem phased in {timer.lap():.2f}s.")
        phased_output_conf(phased, generation)

        # check for convergence
        if settings.pdb_path.is_file() and settings.chk_convergence:
            logger.log(f"checking convergence: FSC calculation")
            fsc = compute_fsc_conf(phased, fsc)
            converge = check_convergence_conf(fsc)
            if converge:
                break
    execution_fence(block=True)

    if settings.must_converge and settings.chk_convergence:
        assert settings.pdb_path.is_file()
        assert converge == True

    logger.log(f"Results saved in {settings.out_dir}")
    logger.log(f"Successfully completed in {timer.total():.2f}s.")


# read the data and run the main algorithm. This can be repeated
@nvtx.annotate("legion/main.py", is_prefix=True)
def main():
    logger = utils.Logger(True, settings)
    logger.log("In Legion main")
    ds = None
    timer = utils.Timer()
    # Reading input images using psana2
    if settings.use_psana:
        (pixel_position, pixel_distance, pixel_index, slices, slices_p) = load_psana()
    # Reading input images from hdf5
    else:
        # Load unique set of intensity slices for python process
        (pixel_position, pixel_distance, pixel_index, slices, slices_p) = get_data(ds)
    logger.log(f"Loaded in {timer.lap():.2f}s.")
    main_task_conf(pixel_position,
                   pixel_distance,
                   pixel_index,
                   slices,
                   slices_p)


