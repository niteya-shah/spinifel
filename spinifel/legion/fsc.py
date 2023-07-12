import numpy as np
import PyNVTX as nvtx
import pygion
import math
from pygion import task, RO, WD, Tunable, Region, execution_fence
from spinifel import settings
from spinifel import utils
from . import utils as lgutils
from eval.fsc import compute_fsc, compute_reference
from eval.align import align_volumes

if settings.use_cupy:
    import cupy

@nvtx.annotate("legion/fsc.py", is_prefix=True)
def create_fsc_regions_multiple():
    fsc_regions = []
    for i in range(settings.N_conformations):
        M = settings.M
        reference = Region((M,M,M), {"ref":pygion.float64},)
        fsc_regions.append(reference)
    return fsc_regions


@nvtx.annotate("legion/fsc.py", is_prefix=True)
def init_fsc(pixel_distance, filename):
    fsc = {}
    dist_recip_max = np.max(pixel_distance.reciprocal)
    reference = compute_reference(filename, settings.M, dist_recip_max)
    fsc["final"] = 0.0
    fsc["delta"] = 1.0
    fsc["res"] = 0.0
    fsc["min_cc"] = settings.fsc_min_cc
    fsc["min_change_cc"] = settings.fsc_min_change_cc
    fsc["dist_recip_max"] = dist_recip_max
    fsc["converge"] = False
    return fsc, reference

@task(leaf=True, privileges=[RO, WD])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/fsc.py", is_prefix=True)
def init_fsc_task(pixel_distance, reference, filename):
    fsc, ref = init_fsc(pixel_distance, filename)
    reference.ref[:] = ref
    return fsc

@task(leaf=True,privileges=[RO("rho_"), RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/fsc.py", is_prefix=True)
def compute_fsc_task(phased, reference, fsc):
    logger = utils.Logger(True,settings)
    if settings.verbosity > 0:
        timer = utils.Timer()

    fsc_dict = fsc
    prev_cc = fsc_dict["final"]
    rho = np.fft.ifftshift(phased.rho_)
    ali_volume, ali_reference, final_cc = align_volumes(
        rho,
        #fsc_dict["reference"],
        reference.ref,
        zoom=settings.fsc_zoom,
        sigma=settings.fsc_sigma,
        n_iterations=settings.fsc_niter,
        n_search=settings.fsc_nsearch,
    )
    resolution, rshell, fsc_val = compute_fsc(
        ali_reference, ali_volume, fsc_dict["dist_recip_max"]
    )
    # uses a lot of memory - release it asap
    if settings.use_cupy and settings.cupy_mempool_clear:
        mempool = cupy.get_default_memory_pool()
        mempool.free_all_blocks()

    if settings.verbosity > 0:
        logger.log(
            f"FSC clear_cupy_mempool:{settings.cupy_mempool_clear} completed in: {timer.lap():.2f}s.",level=1)

    min_cc = fsc_dict["min_cc"]
    delta_cc = final_cc - prev_cc
    min_change_cc = fsc_dict["min_change_cc"]
    fsc_dict["delta"] = delta_cc
    fsc_dict["res"] = resolution
    fsc_dict["final"] = final_cc
    logger.log(
        f"FSC: Check convergence resolution: {resolution:.2f} with cc: {final_cc:.3f} delta_cc:{delta_cc:.5f}.", level=1
    )
    # no change in vals
    if math.isclose(final_cc, min_cc) and math.isclose(delta_cc, min_change_cc):
        return fsc_dict
    else:
        if final_cc > min_cc and delta_cc < min_change_cc:
            fsc_dict["converge"] = True
            logger.log(
                f"Stopping criteria met! Algorithm converged at resolution: {resolution:.2f} with cc: {final_cc:.3f}.",
                level=1)
    return fsc_dict

@nvtx.annotate("legion/fsc.py", is_prefix=True)
def compute_fsc_conf(phased_conf, reference, fsc):
    fsc_dict_array = []
    for i in range(settings.N_conformations):
        # each task returns a future
        # create an array of futures
        if check_convergence_task(fsc[i]).get() is False:
            fsc_dict_val = compute_fsc_task(phased_conf[i], reference[i], fsc[i], point=0)
        else:
            fsc_dict_val = fsc[i]
        fsc_dict_array.append(fsc_dict_val)
    return fsc_dict_array

@task(leaf=True)
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/fsc.py", is_prefix=True)
def check_convergence_task(fsc):
    converge = False
    fsc_dict = fsc
    if fsc_dict["converge"] == True:
        converge = True
    return converge

@nvtx.annotate("legion/fsc.py", is_prefix=True)
def check_convergence_conf(fsc):
    fsc_converge_array = []
    converge = True
    for i in range(settings.N_conformations):
        fsc_converge = check_convergence_task(fsc[i], point=0)
        fsc_converge_array.append(fsc_converge.get())

    for i in range(settings.N_conformations):
        fsc_converge = fsc_converge_array[i]
        if fsc_converge == False:
            converge = False
            return converge
    return converge

# if multiple pdb files exist for multiple conformations
# pdb_path is a directory name else pdb_path is a filename
@nvtx.annotate("legion/fsc.py", is_prefix=True)
def initialize_fsc(pixel_distance):
    fsc = []
    reference = create_fsc_regions_multiple()
    # make sure phased regions are valid
    execution_fence(block=True)
    logger = utils.Logger(True,settings)
    if settings.N_conformations == 1:
        if settings.pdb_path.is_file() and settings.chk_convergence:
            filename = settings.pdb_path
            logger.log(f"fsc filename: {filename}", level=2)
            fsc_future_entry = init_fsc_task(pixel_distance,reference[0],filename, point=0)
            fsc.append(fsc_future_entry)
    else: # multiple conformations
        if settings.pdb_path.is_dir() and settings.chk_convergence:
            num_entries = 0
            for filename in settings.pdb_path.iterdir():
                logger.log(f"fsc filename: {filename}", level=2)
                # an array of futures
                fsc_future_entry = init_fsc_task(pixel_distance,reference[num_entries],filename, point=0)
                fsc.append(fsc_future_entry)
                num_entries = num_entries+ 1
            assert num_entries == settings.N_conformations

        if settings.pdb_path.is_file() and settings.chk_convergence:
            filename = settings.pdb_path
            for i in range(settings.N_conformations):
                fsc_future_entry = init_fsc_task(pixel_distance,reference[i],filename, point=0)
                fsc.append(fsc_future_entry)

    # return a 2 dimension array of fscs
    fsc_all_vals = []
    if len(fsc) > 0:
        for i in range(settings.N_conformations):
            fsc_all_vals.append(fsc[i].get())

    fsc_all = []
    if len(fsc) > 0:
        for i in range(settings.N_conformations):
            fsc_all.append(fsc_all_vals.copy())
    return fsc_all, reference

@nvtx.annotate("legion/fsc.py", is_prefix=True)
def check_convergence_single_conf(fsc):
    fsc_converge_array = []
    converge = False
    for i in range(settings.N_conformations):
        fsc_converge = check_convergence_task(fsc[i], point=0)
        fsc_converge_array.append(fsc_converge)

    # if any of the fsc's have converged - return True
    for i in range(settings.N_conformations):
        fsc_converge = fsc_converge_array[i].get()
        if fsc_converge == True:
            converge = True
            break;
    return converge

# fsc_conv is a multi-dimensional array of futures
# check if all conformations have converged
@nvtx.annotate("legion/fsc.py", is_prefix=True)
def check_convergence_all_conf(fsc_conv):
    for j in range(settings.N_conformations):
        converge = check_convergence_single_conf(fsc_conv[j])
        if converge == False:
            break;
    return converge

@nvtx.annotate("legion/fsc.py", is_prefix=True)
def compute_fsc_single_conf(phased_conf, reference, fsc, conf_id):
    fsc_dict_array = []
    for i in range(settings.N_conformations):
        fsc_dict_val = compute_fsc_task(phased_conf, reference[i], fsc[i], point=0)
        fsc_dict_array.append(fsc_dict_val.get())
    return fsc_dict_array

# check all conformations
@nvtx.annotate("legion/fsc.py", is_prefix=True)
def compute_fsc_conf_all(phased_conf, reference, fsc):
    fsc_dict_array = []
    for i in range(settings.N_conformations):
        if check_convergence_single_conf(fsc[i]) is False:
            fsc_dict_val = compute_fsc_single_conf(phased_conf[i], reference, fsc[i], i)
        else:
            fsc_dict_val = fsc[i]
        fsc_dict_array.append(fsc_dict_val)
    return fsc_dict_array
