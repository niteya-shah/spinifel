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

from .prep import get_data, prep_objects
from .autocorrelation import solve_ac
from .phasing import new_phase, create_phased_regions
from .orientation_matching import match, create_orientations_rp
from . import mapper
from . import checkpoint
from . import utils as lgutils
from .fsc import init_fsc_task, compute_fsc_task, check_convergence_task


@nvtx.annotate("legion/main.py", is_prefix=True)
def load_psana():
    logger = utils.Logger(True)
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


@task(privileges=[RO, RO, RO, RO])
@lgutils.gpu_task_wrapper
def main_task(pixel_position, pixel_distance, pixel_index, slices, slices_p):
    logger = utils.Logger(True)
    timer = utils.Timer()
    curr_gen = 0
    fsc = {}
    total_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    ready_objs = prep_objects(pixel_position, pixel_distance, slices_p, total_procs)

    if settings.pdb_path.is_file() and settings.chk_convergence:
        fsc = init_fsc_task(pixel_distance)
        print(f"initialized FSC", flush=True)
    if settings.load_gen > 0:  # Load input from previous generation
        curr_gen = settings.load_gen
        phased, orientations, orientations_p = checkpoint.load_checkpoint(
            settings.out_dir, settings.load_gen
        )
        phased_region_dict = create_phased_regions(phased)
    else:
        orientations, orientations_p = create_orientations_rp(
            settings.N_images_per_rank
        )
        solved, solve_ac_dict = solve_ac(
            None, 0, pixel_position, pixel_distance, slices_p, ready_objs
        )
        # async tasks logger.log(f"AC recovered in {timer.lap():.2f}s.")

        phased, phased_regions_dict = new_phase(0, solved)
        rho = np.fft.ifftshift(phased.rho_)
        intensity = np.fft.ifftshift(np.abs(np.fft.fftshift(phased.ac) ** 2))
        logger.log(f"Problem phased and AC recovered in {timer.lap():.2f}s.")
        save_mrc(settings.out_dir / f"ac-0.mrc", phased.ac)
        save_mrc(settings.out_dir / f"rho-0.mrc", rho)
        save_mrc(settings.out_dir / f"intensity-0.mrc", intensity)

    curr_gen += 1

    N_generations = settings.N_generations
    for generation in range(curr_gen, N_generations + 1):
        logger.log(f"#" * 27)
        logger.log(f"##### Generation {generation}/{N_generations} #####")
        logger.log(f"#" * 27)

        # Orientation matching
        match(phased, orientations_p, slices_p, settings.N_images_per_rank)

        # Solve autocorrelation
        solved, solve_ac_dict = solve_ac(
            solve_ac_dict,
            generation,
            pixel_position,
            pixel_distance,
            slices_p,
            ready_objs,
            orientations,
            orientations_p,
            phased,
        )

        phased, phased_regions_dict = new_phase(generation, solved, phased_regions_dict)
        # async tasks logger.log(f"Problem phased in {timer.lap():.2f}s.")

        # Check if density converges
        rho = np.fft.ifftshift(phased.rho_)
        intensity = np.fft.ifftshift(np.abs(np.fft.fftshift(phased.ac) ** 2))
        save_mrc(settings.out_dir / f"ac-{generation}.mrc", phased.ac)
        save_mrc(settings.out_dir / f"rho-{generation}.mrc", rho)
        save_mrc(settings.out_dir / f"intensity-{generation}.mrc", intensity)
        logger.log(f"Generation: {generation} completed in {timer.lap():.2f}s.")

        # check for convergence
        if settings.pdb_path.is_file() and settings.chk_convergence:
            print(f"checking convergence: FSC calculation", flush=True)
            fsc = compute_fsc_task(phased, fsc)
            converge = check_convergence_task(fsc)
            converge = converge.get()
            if converge:
                break
    execution_fence(block=True)

    if settings.must_converge:
        assert settings.pdb_path.is_file() and settings.chk_convergence
        assert converge == True

    logger.log(f"Results saved in {settings.out_dir}")
    logger.log(f"Successfully completed in {timer.total():.2f}s.")


# read the data and run the main algorithm. This can be repeated
@nvtx.annotate("legion/main.py", is_prefix=True)
def main():
    logger = utils.Logger(True)
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

    main_task(pixel_position, pixel_distance, pixel_index, slices, slices_p)
