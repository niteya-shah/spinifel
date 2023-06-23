import matplotlib.pyplot as plt
import numpy as np
import skopi as skp
import PyNVTX as nvtx
import pygion
import socket

from pygion import (
    task,
    Future,
    IndexLaunch,
    Partition,
    Region,
    RO,
    WD,
    RW,
    Reduce,
    Tunable,
    execution_fence,
)


from spinifel import settings, utils, image
from . import utils as lgutils
from . import prep as gprep
from scipy.ndimage import gaussian_filter
from .fsc import check_convergence_single_conf
from spinifel.sequential.autocorrelation import ac_with_noise

if settings.use_cupy:
    import os

    os.environ["CUPY_ACCELERATORS"] = "cub"
    from pycuda import gpuarray
    from cupyx.scipy.sparse.linalg import LinearOperator, cg

    # from cupyx.scipy.ndimage import gaussian_filter
    from cupy.linalg import norm
    import cupy as xp
else:
    from scipy.linalg import norm
    from scipy.sparse.linalg import LinearOperator, cg

    xp = np

if settings.use_single_prec:
    f_type = xp.float32
    c_type = xp.complex64
else:
    f_type = xp.float64
    c_type = xp.complex128


@task(leaf=True, privileges=[WD])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def gen_random_orientations(orientations, N_images_per_rank):
    orientations.quaternions[:] = skp.get_random_quat(N_images_per_rank)


@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def get_random_orientations(N_images_per_rank):
    # quaternions are always double precision
    fields_dict = {"quaternions": pygion.float64}
    sec_shape = (4,)
    orientations, orientations_p = lgutils.create_distributed_region(
        N_images_per_rank, fields_dict, sec_shape
    )
    execution_fence(block=True)
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in range(N_procs):
        gen_random_orientations(orientations_p[i], N_images_per_rank, point=i)
    return orientations, orientations_p


@task(leaf=True, privileges=[RO, WD])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def gen_nonuniform_positions_v(nonuniform, nonuniform_v, reciprocal_extent):
    reciprocal_extent = reciprocal_extent.get()
    mult = np.pi / (reciprocal_extent * settings.oversampling)
    nonuniform_v.H[:] = nonuniform.H.reshape(-1) * mult
    nonuniform_v.K[:] = nonuniform.K.reshape(-1) * mult
    nonuniform_v.L[:] = nonuniform.L.reshape(-1) * mult


@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def get_nonuniform_positions_v(
        nonuniform_p, nonuniform_v_p, reciprocal_extent, N_procs,
):
    for i in range(N_procs):
        gen_nonuniform_positions_v(
            nonuniform_p[i], nonuniform_v_p[i], reciprocal_extent, point=i)


@task(leaf=True, privileges=[RO, WD, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def gen_nonuniform_positions(orientations, nonuniform, ready_obj,conf_idx):
    gall = gprep.get_gprep(conf_idx)
    autocorr = gall["mg"]
    (
        nonuniform.H[:],
        nonuniform.K[:],
        nonuniform.L[:],
    ) = autocorr.get_non_uniform_positions(orientations.quaternions)


@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def get_nonuniform_positions(ac_dict, N_procs, ready_objs, conf_idx):
    orientations_p = ac_dict["orientations_p"]
    nonuniform_p = ac_dict["nonuniform_p"]
    for i in range(N_procs):
        gen_nonuniform_positions(
            orientations_p[i], nonuniform_p[i], ready_objs[i], conf_idx, point=i)

# create persistent regions across streams for multiple conformations
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def init_ac_persistent_regions(solve_ac_dict, pixel_position, pixel_distance):
    recip_extent = pixel_distance_rp_max_task(pixel_distance)
    for i in range(settings.N_conformations):
        solve_ac_d = solve_ac_dict[i]
        solve_ac_d["pixel_position"] = pixel_position
        solve_ac_d["pixel_distance"] = pixel_distance
        solve_ac_d["reciprocal_extent"] = recip_extent

# create persistent regions across streams for multiple groups
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def create_solve_regions_multiple():
    N_groups = settings.N_conformations
    solve_regions = []
    for i in range(N_groups):
        solve_regions.append(create_solve_regions_merge())
    return solve_regions

# create persistent regions across streams
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def create_solve_regions_merge():
    solve_dict = {}
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    M = settings.M
    cmpx_type = None
    np_complx_type = None
    if settings.use_single_prec:
        cmpx_type = pygion.complex64
        np_complx_type = (np.complex64,)
        float_type = pygion.float32
    else:
        cmpx_type = pygion.complex128
        np_complx_type = (np.complex128,)
        float_type = pygion.float64

    # ac
    ac = Region((M,) * 3, {"support": pygion.bool_, "estimate": pygion.float64})
    solve_dict["ac"] = ac
    # summary
    summary = Region(
        (N_procs,),
        {
            "rank": pygion.int32,
            "rlambda": pygion.float64,
            "v1": pygion.float64,
            "v2": pygion.float64,
            "image_set": pygion.bool_,
        },
    )
    summary_p = Partition.equal(summary, (N_procs,))
    solve_dict["summary"] = summary
    solve_dict["summary_p"] = summary_p

    results = Region((N_procs * M, M, M), {"ac": pygion.float64})
    results_p = Partition.restrict(results, (N_procs,), [[M], [0], [0]], [M, M, M])
    results_r = Region((M, M, M), {"ac": pygion.float64})
    solve_dict["results"] = results
    solve_dict["results_p"] = results_p
    solve_dict["results_r"] = results_r
    # create a dictionary of regions/partitions
    return solve_dict


@task(leaf=True, privileges=[RO("ac"), WD("support", "estimate")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def phased_to_constrains(phased, ac):
    ac_smoothed = gaussian_filter(phased.ac, 0.5)
    ac.support[:] = ac_smoothed > 1e-12
    ac.estimate[:] = phased.ac * ac.support

@task(leaf=True, privileges=[RO, WD, WD, RO, RO, RO, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def solve_simple(
    ac,
    result,
    summary,
    conf,
    ready_obj,
    orientations,
    slices,
    M,
    M_ups,
    Mtot,
    generation,
    rank,
    alambda,
    rlambda,
    flambda,
    use_reciprocal_symmetry,
    group_idx,
    n_conf
    ):
    mg = gprep.get_gprep(group_idx)["mg"]
    logger = gprep.get_gprep(group_idx)["logger"]

    N_images_per_rank = slices.ispace.domain.extent[0]
    conf_local = conf.conf_id
    conf_local = conf_local[group_idx*N_images_per_rank:group_idx*N_images_per_rank+N_images_per_rank]

    num_images = np.sum(conf_local, dtype=np.int64)
    logger.log(f"started solve:[n_conf,conf_index]: [{n_conf},{group_idx}],  conf_shape: {conf.conf_id.shape}, conf_dtype: {conf.conf_id.dtype}, num_images={num_images}", level=2)
    # initialize image_set field to False and update summary to
    # default values and return
    if num_images == 0:
        summary.rank[0] = rank
        summary.rlambda[0] = rlambda
        summary.v1[0] = 1000.0
        summary.v2[0] = 1000.0
        summary.image_set[0] = False
        result.ac[:] = 1.0
    else:
        ret,W,d = mg.solve_ac_common(slices.data, orientations.quaternions, ac.estimate,
                                     ac.support, conf_local, rlambda, flambda)
        if not isinstance(ret, np.ndarray):
            ac_res = ret.get()
        else:
            ac_res = ret;
        ac_res = ac_res.reshape((M,) * 3)
        if use_reciprocal_symmetry:
            assert np.all(np.isreal(ac_res))
        ac_res = ac_res.real
        result.ac[:] = np.ascontiguousarray(ac_res)
        image.show_volume(
            ac_res, settings.Mquat, f"autocorrelation_conf_{group_idx}_{generation}_{rank}.png"
        )
        v1 = norm(ret)
        v2 = norm(W * ret - d)
        if not isinstance(v1, np.ndarray) and not isinstance(v1, float):
            v1 = v1.get()
            v2 = v2.get()
        summary.rank[0] = rank
        summary.rlambda[0] = rlambda
        summary.v1[0] = v1
        summary.v2[0] = v2
        summary.image_set[0] = True

@task(leaf=True, privileges=[None, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def check_multiple_conf_summary(generation, summary):
    # generation 0 will always have non-empty diffraction patterns
    # if any summary result has non-empty differaction patterns
    any_val = np.any(summary.image_set)
    if generation == 0 or any_val:
        return True
    # we have empty diffraction patterns from all ranks for this
    # conformation
    else:
        return False

# copy ac with noise
# this needs to be updated
@task(leaf=True, privileges=[RW, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def copy_ac_with_noise(results_p0, results_conf, copy_status_dst, copy_status_src, src_id, dst_id):
    # do we need to copy and is this a valid 'ac' to copy from?
    if copy_status_dst.get() is not True and copy_status_src.get() is True:
        logger = utils.Logger(True, settings)
        logger.log(f"copying AC with noise for conformation {dst_id} from conformation {src_id}", level=2)
        results_p0.ac[:] = ac_with_noise(results_conf.ac)
        return True
    # if we don't need to copy
    elif copy_status_dst.get() is True:
        return True
    else:
        # we didn't copy and we still need to find a valid 'ac'
        return False

@task(leaf=True, privileges=[None, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def select_ac(generation, summary, conf_ok):
    iref = 0
    if conf_ok.get() is not True:
        return -1
    else:
        if generation == 0:
            # Heuristic: retain rank with highest lambda and high v1.
            idx = summary.v1 >= np.mean(summary.v1)
            imax = np.argmax(summary.rlambda[idx])
            iref = np.arange(len(summary.rank), dtype=np.int)[idx][imax]
        else:
            # deal with ranks that don't have any diffraction patterns due to
            # multiple conformations
            # discard the rank(s) with empty diffraction patterns
            max_val = np.max(summary.v1 + summary.v2)
            vals = np.where(summary.image_set is True, summary.v1 + summary.v2, max_val)
            # Take corner of L-curve: min (v1+v2)
            iref = np.argmin(summary.v1 + summary.v2)
        return iref


@task(leaf=True, privileges=[WD, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def ac_result_subregion(results_p0, results):
    results_p0.ac[:] = results.ac[:]


# partition the region and use only the iref.get() subregion
@task(inner=True, privileges=[WD, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def ac_result_task(results_p0, results, results_p, iref, group_idx):
    indx = iref.get()
    # initialize the region - it will use a copy with noise
    if indx == -1:
        pygion.fill(results_p0, "ac", 0.0)
    else:
        ac_result_subregion(results_p0, results_p[indx], point=group_idx)


@task(leaf=True, privileges=[RO("reciprocal")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def pixel_distance_rp_max_task(pixel_distance):
    return pixel_distance.reciprocal.max()


@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def solve_ac_merge(
        solve_ac_dict,
        generation,
        pixel_position,
        pixel_distance,
        slices_p,
        ready_objs,
        conf_p,
        group_idx,
        orientations=None,
        orientations_p=None,
        phased=None,
        str_mode=False
):
    M = settings.M
    M_ups = settings.M_ups  # For upsampled convolution technique
    Mtot = M**3
    N_images_per_rank = slices_p[0].ispace.domain.extent[0]
    N = N_images_per_rank * utils.prod(settings.reduced_det_shape)
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    use_reciprocal_symmetry = True
    maxiter = settings.solve_ac_maxiter
    fill_orientations = False

    if str_mode == True:
        solve_ac_dict["slices_p"] = slices_p
        solve_ac_dict["ready_objs"] = ready_objs
        if orientations is None:
            orientations, orientations_p = get_random_orientations(N_images_per_rank)
        solve_ac_dict["orientations"] = orientations
        solve_ac_dict["orientations_p"] = orientations_p
    elif orientations is None:
        orientations, orientations_p = get_random_orientations(N_images_per_rank)
        solve_ac_dict = create_solve_regions_merge();
        solve_ac_dict["reciprocal_extent"] = pixel_distance_rp_max_task(pixel_distance)
        solve_ac_dict["orientations"] = orientations
        solve_ac_dict["orientations_p"] = orientations_p
        solve_ac_dict["pixel_position"] = pixel_position
        solve_ac_dict["pixel_distance"] = pixel_distance
        solve_ac_dict["slices_p"] = slices_p
        solve_ac_dict["ready_objs"] = ready_objs
    else:
        solve_ac_dict["orientations"] = orientations
        solve_ac_dict["orientations_p"] = orientations_p

    ac = solve_ac_dict["ac"]
    if phased is None:
        pygion.fill(ac, "support", 1)
        pygion.fill(ac, "estimate", 0.0)
    else:
        phased_to_constrains(phased, ac)

    weights = 1
    results = solve_ac_dict["results"]
    results_p = solve_ac_dict["results_p"]
    results_r = solve_ac_dict["results_r"]

    alambda = 1
    rlambdas = Mtot / N * 2 ** (np.arange(N_procs) - N_procs / 2).astype(np.float)
    flambdas = 1e5 * 10 ** (np.arange(N_procs) - N_procs // 2).astype(np.float)
    summary_p = solve_ac_dict["summary_p"]
    summary = solve_ac_dict["summary"]
    reciprocal_extent = solve_ac_dict["reciprocal_extent"]
    for i in range(N_procs):
        solve_simple(
            ac,
            results_p[i],
            summary_p[i],
            conf_p[i],
            ready_objs[i],
            orientations_p[i],
            slices_p[i],
            M,
            M_ups,
            Mtot,
            generation,
            i,
            alambda,
            rlambdas[i],
            flambdas[i],
            use_reciprocal_symmetry,
            group_idx,
            settings.N_conformations,
            point=i)

    # check if all ranks for this conformation
    conf_ok = check_multiple_conf_summary(generation, summary)
    iref = select_ac(generation, summary, conf_ok)
    # remove blocking call
    # return results_p[iref.get()], solve_ac_dict
    ac_result_task(results_r, results, results_p, iref, group_idx, point=group_idx)
    return results_r, solve_ac_dict, conf_ok

# phased is an array of regions
# orientations is an array of regions
# orientations_p is an array of partitions
# read_objs is an array of ready_objs
# solve_ac_dict is an array of solve_ac dictionaries
# conf_p is a region/partition that contains the result from orientation
# matching -> percentage of min_dist for each conformation and each
# diffraction image -> [N_images_per_rank, N_conformations]
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def solve_ac_conf(
    solve_ac_dict,
    generation,
    pixel_position,
    pixel_distance,
    slices_p,
    ready_objs,
    conf_p,
    fsc,
    orientations=None,
    orientations_p=None,
    phased=None,
    str_mode=False,
):
    create_regions = False
    # str_mode = streaming mode
    # solve ac dictionary is created at the start
    if phased is None and not str_mode:
        create_regions = True
        assert solve_ac_dict is None
        assert orientations is None
        assert orientations_p is None
    solve_ac_array = []
    result_array = []
    conf_ok_array = []
    logger = utils.Logger(True, settings)
    for i in range(settings.N_conformations):
        # check if converged
        if len(fsc) > 0 and check_convergence_single_conf(fsc[i]):
            logger.log(f"conformation {i} HAS converged in solve_ac")
            assert create_regions is False
            results = solve_ac_dict[i]["results_r"]
            result_array.append(results)
            conf_ok_array.append(Future(True, pygion.bool_))
        else:
            if len(fsc) > 0:
                logger.log(f"conformation {i} has NOT converged in solve_ac")
            if str_mode:
                if orientations is not None:
                    results, solve_ac_dict[i], conf_ok = solve_ac_merge(solve_ac_dict[i],
                                                               generation, pixel_position,
                                                               pixel_distance,
                                                               slices_p, ready_objs, conf_p, i,
                                                               orientations[i], orientations_p[i],
                                                               phased[i], str_mode)
                    result_array.append(results)
                    conf_ok_array.append(conf_ok)
                else:
                    results, solve_ac_dict[i], conf_ok = solve_ac_merge(solve_ac_dict[i],
                                                               generation, pixel_position,
                                                               pixel_distance,
                                                               slices_p, ready_objs, conf_p, i,
                                                               None, None,
                                                               None, str_mode)
                    result_array.append(results)
                    conf_ok_array.append(conf_ok)
            else:
                if create_regions == False:
                    results, solve_ac_dict[i], conf_ok = solve_ac_merge(solve_ac_dict[i],
                                                               generation, pixel_position,
                                                               pixel_distance,
                                                               slices_p, ready_objs, conf_p, i,
                                                               orientations[i], orientations_p[i],
                                                               phased[i], str_mode)
                    result_array.append(results)
                    conf_ok_array.append(conf_ok)
                else:
                    results, solve_ac_dict_entry, conf_ok = solve_ac_merge(solve_ac_dict, generation, pixel_position,
                                                                  pixel_distance,
                                                                  slices_p, ready_objs, conf_p, i,
                                                                  orientations, orientations_p, phased, str_mode)
                    solve_ac_array.append(solve_ac_dict_entry)
                    result_array.append(results)
                    conf_ok_array.append(conf_ok)

    # handle the case of empty diffraction images in a particular conformation
    # copy 'ac' with noise from another
    if settings.N_conformations > 1:
        for i in range(settings.N_conformations):
            for j in range(settings.N_conformations):
                if i != j:
                    conf_ok_array[i] = copy_ac_with_noise(result_array[i], result_array[j], conf_ok_array[i], conf_ok_array[j], j, i)

    if create_regions:
        return result_array, solve_ac_array

    return result_array, solve_ac_dict
