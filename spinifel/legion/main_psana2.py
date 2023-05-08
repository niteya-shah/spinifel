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

from .prep import (
    get_data,
    init_partitions_regions_psana2,
    load_image_batch,
    load_pixel_data,
    process_data,
    prep_objects_multiple,
)
from .utils import union_partitions_with_stride, fill_region, dump_single_partition
from .autocorrelation import (
    solve_ac_conf,
    create_solve_regions_multiple,
    get_random_orientations,
    pixel_distance_rp_max_task,
    init_ac_persistent_regions,

)
from .phasing import (
    phased_output_conf,
    new_phase_conf,
    create_phase_regions_multiple,
)

from .orientation_matching import (
    match_conf,
    create_orientations_rp,
    create_min_dist_rp,
)

from . import mapper
from . import checkpoint
from . import utils as lgutils
from .fsc import init_fsc_task, compute_fsc_conf, check_convergence_conf

@nvtx.annotate("legion/main.py", is_prefix=True)
def load_psana():
    logger = utils.Logger(True, settings)
    assert settings.use_psana == True
    # Reading input images using psana2
    # assert settings.ps_smd_n_events == settings.N_images_per_rank
    # For now, we use one smd chunk per node just to keep things simple.
    # parameters used
    # settings.ps_batch_size -> minimum batch size
    # settings.
    # print(f"setting.ps_smd_n_events={settings.ps_smd_n_events} N_images_per_rank={settings.N_images_per_rank}", flush=True)
    assert settings.ps_smd_n_events == settings.N_images_per_rank
    from psana.psexp.tools import mode
    from psana import DataSource

    total_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    # minimum images to load per batch
    min_batch_size = settings.N_images_per_rank

    # for now assume N_images_max is a multiple of N_images_per_rank
    assert settings.N_images_max % min_batch_size == 0

    max_batches = settings.N_images_max // min_batch_size

    # preload a max of N_image_batches_max per iteration
    max_batches_per_iter = settings.N_image_batches_max

    # max_images_per_iter
    max_images_per_iter = settings.N_image_batches_max * min_batch_size

    logger.log(
        f"Using psana: exp={settings.ps_exp}, run={settings.ps_runnum}, dir={settings.ps_dir}, max_batches_per_iter={max_batches_per_iter}, max_batches={max_batches}, mode={mode}"
    )
    assert mode == "legion"
    max_events = settings.N_images_max * total_procs
    # todo add run array settings
    ds = DataSource(
        exp=settings.ps_exp,
        run=settings.ps_runnum,
        dir=settings.ps_dir,
        max_events=max_events,
    )

    (
        slices,
        all_partitions,
        slices_images,
        slices_images_p,
    ) = init_partitions_regions_psana2()

    # load pixel_position, pixel_distance, pixel_index
    pixel_position, pixel_distance, pixel_index, run = load_pixel_data(ds)
    gen_run = ds.runs()
    gen_run, gen_smd, run = load_image_batch(run, gen_run, None, slices_images_p[0])

    pixel_position, pixel_distance, pixel_index = process_data(
        slices_images,
        slices_images_p[0],
        slices,
        all_partitions[0],
        pixel_distance,
        pixel_index,
        pixel_position,
        0,
    )

    return (
        pixel_position,
        pixel_distance,
        pixel_index,
        slices,
        all_partitions[0],
        all_partitions,
        gen_run,
        gen_smd,
        run,
        slices_images,
        slices_images_p,
    )


def load_psana_subset(
    gen_run,
    gen_smd,
    batch_size,
    cur_batch_size,
    slices,
    all_partitions,
    run,
    slices_images,
    slices_images_all_p,
    idx,
    pixel_position,
    pixel_distance,
    pixel_index,
):

    # cur_batch_size  must be a multiple of batch_size
    assert cur_batch_size % batch_size == 0
    slices_p = all_partitions[cur_batch_size // batch_size]
    slices_images_p = slices_images_all_p[idx]

    gen_run, gen_smd, run = load_image_batch(run, gen_run, gen_smd, slices_images_p)

    # bin data
    pixel_position, pixel_distance, pixel_index = process_data(
        slices_images,
        slices_images_p,
        slices,
        slices_p,
        pixel_distance,
        pixel_index,
        pixel_position,
        idx,
    )

    return slices_p, gen_run, gen_smd, run, pixel_distance, pixel_index, pixel_position


def main_spinifel(
    pixel_position,
    pixel_distance,
    pixel_index,
    slices,
    slices_p,
    n_images_per_rank,
    solve_ac_dict,
    start_gen,
    end_gen,
    phased_regions_dict,
    fsc_array,
    ready_objs_p
):

    logger = utils.Logger(True, settings)
    timer = utils.Timer()
    curr_gen = 0
    total_procs = Tunable.select(Tunable.GLOBAL_PYS).get()

    prep_objects_multiple(pixel_position, pixel_distance, slices_p, ready_objs_p, total_procs)

    # regions specific to to multiple conformations
    min_dist = None
    min_dist_p = None
    min_dist_proc = None
    conf = None
    conf_p = None
    fsc = fsc_array

    # regions/partitions related to multiple conformations
    # based on n_images_per_rank

    conf_regions_dict = {}
    conf_regions_dict = create_min_dist_rp(n_images_per_rank,
                                           settings.N_conformations)
    min_dist =  conf_regions_dict["min_dist"]
    min_dist_p = conf_regions_dict["min_dist_p"]
    min_dist_proc = conf_regions_dict["min_dist_proc"]
    conf = conf_regions_dict["conf"]
    conf_p = conf_regions_dict["conf_p"]

    # regions related to phasing for multiple conformations
    phased = []
    for i in range(settings.N_conformations):
        phased.append(phased_regions_dict[i]["phased"])

    # not supported for streaming/psana mode since image data grows
    assert settings.load_gen == -1

    # orientations for each conformation
    orientations_a = []
    orientations_a_p = []
    phased = []
    if start_gen == 0:
        solved, solve_ac_dict = solve_ac_conf(
            solve_ac_dict,
            0,
            pixel_position,
            pixel_distance,
            slices_p,
            ready_objs_p,
            conf_p,
            fsc,
            None,
            None,
            None,
            True,
        )

        phased, phased_regions_dict = new_phase_conf(
            start_gen, solved, fsc, phased_regions_dict
        )
        phased_output_conf(phased, start_gen)

        for i in range(settings.N_conformations):
            orientations_a.append(solve_ac_dict[i]["orientations"])
            orientations_a_p.append(solve_ac_dict[i]["orientations_p"])

        # make sure all partitions are valid
        execution_fence(block=True)
    else:  # streaming
        # setup non-persistent  regions/partitions per conformation
        # i.e. those that are dependent on number of images per rank
        # or array of conformations
        for i in range(settings.N_conformations):
            phased.append(phased_regions_dict[i]["phased"])

            # setup new regions based on n_images_per_rank
            orientations, orientations_p = get_random_orientations(n_images_per_rank)
            orientations_a.append(orientations)
            orientations_a_p.append(orientations_p)
            solve_ac_dict[i]["orientations"] = orientations
            solve_ac_dict[i]["orientations_p"] = orientations_p
            solve_ac_dict[i]["slices_p"] = slices_p
            solve_ac_dict[i]["ready_objs"] = ready_objs_p
        # make sure all partitions are valid
        execution_fence(block=True)

    curr_gen = curr_gen + 1 + start_gen
    N_generations = end_gen
    N_gens_stream = settings.N_gens_stream
    assert N_gens_stream <= N_generations
    for generation in range(curr_gen, N_generations + 1):
        logger.log(f"#" * 27)
        logger.log(f"##### Generation {generation}/{N_generations} #####")
        logger.log(f"#" * 27)

        match_conf(phased, orientations_a_p, slices_p, min_dist_p, min_dist_proc, conf_p, n_images_per_rank, ready_objs_p, fsc)
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
            orientations_a,
            orientations_a_p,
            phased,
        )
        phased, phased_regions_dict = new_phase_conf(
            generation, solved, fsc, phased_regions_dict
        )
        phased_output_conf(phased, generation)

        if settings.pdb_path.is_file() and settings.chk_convergence:
            fsc = compute_fsc_conf(phased, fsc)
            converge = check_convergence_conf(fsc)
            if converge:
                break

    # fill regions so they get garbage collected
    # regions are based on number of images per rank
    for i in range(settings.N_conformations):
        fill_region(orientations_a[i], 0)
    fill_region(min_dist, 0.0)
    fill_region(conf, 0)

# read the data and run the main algorithm. This can be repeated
@nvtx.annotate("legion/main.py", is_prefix=True)
def main():
    logger = utils.Logger(True, settings)
    logger.log("In Legion Psana2 with Streaming")
    ds = None
    timer = utils.Timer()
    # Reading input images using psana2
    assert settings.use_psana
    # compute pixel_position, pixel_distance, pixel_index, partitions,
    # regions
    (
        pixel_position,
        pixel_distance,
        pixel_index,
        slices,
        slices_p,
        all_partitions,
        gen_run,
        gen_smd,
        run,
        slices_images,
        slices_images_p,
    ) = load_psana()
    logger.log(f"Loaded in {timer.lap():.2f}s.")
    done = False
    batch_size = settings.N_images_per_rank
    cur_batch_size = batch_size
    max_batch_size = settings.N_images_max
    n_points = Tunable.select(Tunable.GLOBAL_PYS).get()
    solve_ac_dict = None

    # reuse across stream
    phased_regions_dict = create_phase_regions_multiple()

    # reuse regions across streams
    solve_ac_dict = create_solve_regions_multiple()

    # reuse dictionary items across streams
    init_ac_persistent_regions(solve_ac_dict, pixel_position, pixel_distance)

    # reuse ready_objs_p across streams
    ready_objs, ready_objs_p = lgutils.create_distributed_region(
        Tunable.select(Tunable.GLOBAL_PYS).get(), {"done": pygion.bool_}, ()
    )

    # initialize fsc per conformation
    fsc = []
    if settings.chk_convergence and settings.pdb_path.is_file():
        for i in range(settings.N_conformations):
            fsc_future_entry = init_fsc_task(pixel_distance)
            fsc.append(fsc_future_entry)
        logger.log(f"initialized FSC", level=1)

    N_generations = settings.N_generations
    N_gens_stream = settings.N_gens_stream
    assert N_gens_stream <= N_generations
    start_gen = 0
    end_gen = N_gens_stream
    while not done:
        # slices_p reflects union of new images that have been loaded
        # and earlier images in the previous partition
        if cur_batch_size != batch_size:
            slices_p = union_partitions_with_stride(
                slices,
                settings.reduced_det_shape,
                cur_batch_size,
                max_batch_size,
                max_batch_size,
                n_points,
            )
        if cur_batch_size == max_batch_size:
            end_gen = N_generations
        logger.log(f"cur_batch_size = {cur_batch_size}, batch_size = {batch_size}, N_image_batches_max={settings.N_image_batches_max}, max_batch_size={max_batch_size}, start_gen={start_gen}, end_gen={end_gen}",level=settings.verbosity)
        execution_fence(block=True)
        main_spinifel(
            pixel_position,
            pixel_distance,
            pixel_index,
            slices,
            slices_p,
            cur_batch_size,
            solve_ac_dict,
            start_gen,
            end_gen,
            phased_regions_dict,
            fsc,
            ready_objs_p
        )
        start_gen = end_gen
        end_gen = min(N_generations, end_gen + N_gens_stream)
        if end_gen > N_generations:
            end_gen = N_generations
        if cur_batch_size == max_batch_size:
            done = True
        if not done:
            for i in range(settings.N_image_batches_max):
                (
                    slices_p,
                    gen_run,
                    gen_smd,
                    run,
                    pixel_distance,
                    pixel_index,
                    pixel_position,
                ) = load_psana_subset(
                    gen_run,
                    gen_smd,
                    batch_size,
                    cur_batch_size,
                    slices,
                    all_partitions,
                    run,
                    slices_images,
                    slices_images_p,
                    i,
                    pixel_position,
                    pixel_distance,
                    pixel_index,
                )
                cur_batch_size = cur_batch_size + batch_size
                if cur_batch_size == max_batch_size:
                    break

    execution_fence(block=True)
    logger.log(f"Results saved in {settings.out_dir}")
    logger.log(f"Successfully completed in {timer.total():.2f}s.")
