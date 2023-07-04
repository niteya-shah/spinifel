import matplotlib.pyplot as plt
import numpy as np
import skopi as skp
import PyNVTX as nvtx
import pygion
import socket

from pygion import (
    task,
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
from .fsc import check_convergence_task

if settings.use_cupy:
    import os

    os.environ["CUPY_ACCELERATORS"] = "cub"
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


# equivalent to setup_linops
@task(privileges=[RO, WD, RO, RO, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def right_hand_ADb_task(
        slices, uregion, nonuniform_v, ac, ready_obj, M, use_reciprocal_symmetry,conf_idx=0):
    logger = gprep.get_gprep(conf_idx)["logger"]
    logger.log(f"{socket.gethostname()} started ADb.",level=1)
    autocorr = gprep.get_gprep(conf_idx)["mg"]
    ac_support = xp.array(ac.support)
    adj = autocorr.nufft.adjoint(
        autocorr.nuvect_Db,
        nonuniform_v.H,
        nonuniform_v.K,
        nonuniform_v.L,
        ac_support,
        use_reciprocal_symmetry,
        M,
    )
    if not isinstance(adj, np.ndarray):
        adj = adj.get()

    uregion.ADb[:] = adj
    logger.log(f"{socket.gethostname()} computed ADb.",level=1)


@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def right_hand(
        slices_p, uregion_p, nonuniform_v_p, ac, M, use_reciprocal_symmetry, ready_objs, group_idx):
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()

    for i in range(N_procs):
        right_hand_ADb_task(
            slices_p[i],
            uregion_p[i],
            nonuniform_v_p[i],
            ac,
            ready_objs[i],
            M,
            use_reciprocal_symmetry,
            group_idx,
            point=i)


@task(leaf=True, privileges=[WD, RO, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def prep_Fconv_task(
        uregion_ups,
        nonuniform_v,
        ready_obj,
        weights,
        M_ups,
        Mtot,
        N,
        reciprocal_extent,
        use_reciprocal_symmetry,
        conf_idx=0
):

    logger = gprep.get_gprep(conf_idx)["logger"]
    logger.log(f"{socket.gethostname()} started Fconv.", level=1)
    autocorr = gprep.get_gprep(conf_idx)["mg"]
    conv_ups = autocorr.nufft.adjoint(
        autocorr.nuvect,
        nonuniform_v.H,
        nonuniform_v.K,
        nonuniform_v.L,
        1,
        use_reciprocal_symmetry,
        M_ups,
    )
    f_conv = xp.fft.fftn(xp.fft.ifftshift(xp.array(conv_ups)))
    if not isinstance(f_conv, np.ndarray):
        f_conv = f_conv.get()
    uregion_ups.F_conv_[:] = f_conv
    logger.log(f"{socket.gethostname()} computed Fconv.", level=1)


@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def prep_Fconv(
    uregion_ups,
    nonuniform_v,
    nonuniform_v_p,
    weights,
    M_ups,
    Mtot,
    N,
    reciprocal_extent,
    use_reciprocal_symmetry,
    ready_objs,
    group_idx,
):
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in range(N_procs):
        prep_Fconv_task(
            uregion_ups[i],
            nonuniform_v_p[i],
            ready_objs[i],
            weights,
            M_ups,
            Mtot,
            N,
            reciprocal_extent,
            use_reciprocal_symmetry,
            point=i)


@task(leaf=True, privileges=[WD, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def prep_nuvect(nuvect, slices, N, f_type):
    data = slices.data.reshape(-1)
    weights = np.ones(N, dtype=f_type)
    nuvect.nuvect_Db[:] = data * weights
    nuvect.nuvect[:] = np.ones_like(data)


# needed only once for all generations
@task(leaf=True, privileges=[WD("F_antisupport")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def prep_Fantisupport(uregion, M):
    # lu = np.linspace(-np.pi, np.pi, M)
    # Use integers instead of reals in order to avoid roundoff peculiarities.
    Mhalf = (M-1)//2
    iu = range(-Mhalf, Mhalf+1)
    # Hu_, Ku_, Lu_ = np.meshgrid(lu, lu, lu, indexing="ij")
    Hi_, Ki_, Li_ = np.meshgrid(iu, iu, iu, indexing="ij")
    # Qu_ = np.around(np.sqrt(Hu_**2 + Ku_**2 + Lu_**2), 4)
    Qi2_ = Hi_**2 + Ki_**2 + Li_**2
    # uregion.F_antisupport[:] = Qu_ > np.pi / settings.oversampling
    uregion.F_antisupport[:] = (Qi2_ * settings.oversampling**2 > Mhalf**2)
    # Generate an antisupport in Fourier space, which has zeros in the central
    # sphere and ones in the high-resolution corners.
    Fantisup = uregion.F_antisupport
    assert np.all(Fantisup[:] == Fantisup[::-1, :, :])
    assert np.all(Fantisup[:] == Fantisup[:, ::-1, :])
    assert np.all(Fantisup[:] == Fantisup[:, :, ::-1])
    assert np.all(Fantisup[:] == Fantisup[::-1, ::-1, ::-1])


# garbage collect all autocorrelation regions
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def fill_autocorrelation_regions(solve_dict, persist=False):
    if solve_dict is None:
        return
    # garbage collect all regions or non-persistent regions
    if persist == False:
        if "uregion" in solve_dict:
            pygion.fill(solve_dict["uregion"], "ADb", 0.0)
            pygion.fill(solve_dict["uregion"], "F_antisupport", True)
        if "uregion_ups" in solve_dict:
            lgutils.fill_region_task(solve_dict["uregion_ups"], complex(0, 0))
        if "ac" in solve_dict:
            pygion.fill(solve_dict["ac"], "support", True)
            pygion.fill(solve_dict["ac"], "estimate", 0.0)
        if "summary" in solve_dict:
            pygion.fill(solve_dict["summary"], "rank", 0)
            pygion.fill(solve_dict["summary"], "rlambda", 0.0)
            pygion.fill(solve_dict["summary"], "v1", 0.0)
            pygion.fill(solve_dict["summary"], "v2", 0.0)
        if "results" in solve_dict:
            pygion.fill(solve_dict["results"], "ac", 0.0)
        if "results_r" in solve_dict:
            pygion.fill(solve_dict["results_r"], "ac", 0.0)

    if "nonuniform_v" in solve_dict:
        lgutils.fill_region_task(solve_dict["nonuniform_v"], 0.0)
    if "nonuniform" in solve_dict:
        lgutils.fill_region_task(solve_dict["nonuniform"], 0.0)


# create all the region
# initialize regions
# str=True in streaming mode
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def prepare_solve_all_gens(slices_p, solve_dict, str_mode=False):

    if solve_dict is None:
        solve_dict = {}
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    N_images_per_rank = slices_p[0].ispace.domain.extent[0]
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

    N_vals_per_rank = N_images_per_rank * utils.prod(settings.reduced_det_shape)

    if str_mode == False:
        fields_dict = {"ADb": float_type, "F_antisupport": pygion.bool_}
        uregion, uregion_p = lgutils.create_distributed_region(
            M,
            fields_dict,
            (
                M,
                M,
            ),
        )
        solve_dict["uregion"] = uregion
        solve_dict["uregion_p"] = uregion_p
        # For upsampled convolution technique
        M_ups = settings.M_ups
        fields_dict = {"F_conv_": cmpx_type}
        uregion_ups, uregion_ups_p = lgutils.create_distributed_region(
            M_ups,
            fields_dict,
            (
                M_ups,
                M_ups,
            ),
        )
        solve_dict["uregion_ups"] = uregion_ups
        solve_dict["uregion_ups_p"] = uregion_ups_p
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

    # H, K, L
    fields_dict = {"H": pygion.float64, "K": pygion.float64, "L": pygion.float64}
    # nonuniform_v
    sec_shape = ()
    nonuniform_v, nonuniform_v_p = lgutils.create_distributed_region(
        N_vals_per_rank, fields_dict, sec_shape
    )
    if "nonuniform_v" in solve_dict:
        lgutils.fill_region_task(solve_dict["nonuniform_v"], 0.0)
    solve_dict["nonuniform_v"] = nonuniform_v
    solve_dict["nonuniform_v_p"] = nonuniform_v_p

    # nonuniform
    sec_shape = settings.reduced_det_shape
    fields_dict = {"H": float_type, "K": float_type, "L": float_type}
    nonuniform, nonuniform_p = lgutils.create_distributed_region(
        N_images_per_rank, fields_dict, sec_shape
    )

    if "nonuniform" in solve_dict:
        lgutils.fill_region_task(solve_dict["nonuniform"], 0.0)
    solve_dict["nonuniform"] = nonuniform
    solve_dict["nonuniform_p"] = nonuniform_p

    # create a dictionary of regions/partitions
    return solve_dict


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
        solve_regions.append(create_solve_regions())
    return solve_regions

# create persistent regions across streams
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def create_solve_regions():
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

    fields_dict = {"ADb": float_type, "F_antisupport": pygion.bool_}
    uregion, uregion_p = lgutils.create_distributed_region(
        M,
        fields_dict,
        (
            M,
            M,
        ),
    )
    solve_dict["uregion"] = uregion
    solve_dict["uregion_p"] = uregion_p

    # For upsampled convolution technique
    M_ups = settings.M_ups
    fields_dict = {"F_conv_": cmpx_type}
    uregion_ups, uregion_ups_p = lgutils.create_distributed_region(
        M_ups,
        fields_dict,
        (
            M_ups,
            M_ups,
        ),
    )

    if "uregion_ups" in solve_dict:
        lgutils.fill_region_task(solve_dict["uregion_ups"], complex(0, 0))

    solve_dict["uregion_ups"] = uregion_ups
    solve_dict["uregion_ups_p"] = uregion_ups_p
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


@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def prepare_solve(
        solve_ac_dict, slices_p, weights, M, Mtot, M_ups, N, use_reciprocal_symmetry, group_idx
):

    N_images_per_rank = slices_p[0].ispace.domain.extent[0]
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    ac = solve_ac_dict["ac"]
    nonuniform = solve_ac_dict["nonuniform"]
    nonuniform_p = solve_ac_dict["nonuniform_p"]
    nonuniform_v = solve_ac_dict["nonuniform_v"]
    nonuniform_v_p = solve_ac_dict["nonuniform_v_p"]
    reciprocal_extent = solve_ac_dict["reciprocal_extent"]
    uregion_ups_p = solve_ac_dict["uregion_ups_p"]
    uregion_p = solve_ac_dict["uregion_p"]
    ready_objs = solve_ac_dict["ready_objs"]
    get_nonuniform_positions_v(nonuniform_p, nonuniform_v_p, reciprocal_extent, N_procs)

    prep_Fconv(
        uregion_ups_p,
        nonuniform_v,
        nonuniform_v_p,
        weights,
        M_ups,
        Mtot,
        N,
        reciprocal_extent,
        use_reciprocal_symmetry,
        ready_objs,
        group_idx,
    )

    right_hand(
        slices_p, uregion_p, nonuniform_v_p, ac, M, use_reciprocal_symmetry, ready_objs, group_idx
    )


@task(leaf=True, privileges=[RO("ac"), WD("support", "estimate")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def phased_to_constrains(phased, ac):
    ac_smoothed = gaussian_filter(phased.ac, 0.5)
    ac.support[:] = ac_smoothed > 1e-12
    ac.estimate[:] = phased.ac * ac.support



@task(leaf=True, privileges=[RO, RO, RO, WD, WD, RO, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def solve(
    uregion,
    uregion_ups,
    ac,
    result,
    summary,
    conf,
    ready_obj,
    M,
    M_ups,
    Mtot,
    generation,
    rank,
    alambda,
    rlambda,
    flambda,
    reciprocal_extent,
    use_reciprocal_symmetry,
    maxiter,
    group_idx,
    n_conf
):
    logger = gprep.get_gprep(group_idx)["logger"]
    logger.log(f"started solve:[n_conf,conf_index]: [{n_conf},{group_idx}],  conf_shape: {conf.conf_id.shape}, conf_dtype: {conf.conf_id.dtype}", level=2)
    logger.log(f"conf_id: {conf.conf_id}", level=2)

    def W_matvec(uvect):
        """Define W part of the W @ x = d problem."""
        uvect_ADA = gprep.get_gprep(group_idx)["mg"].core_problem_convolution(
            uvect, xp.array(uregion_ups.F_conv_), xp.array(ac.support)
        )
        uvect_FDF = gprep.get_gprep(group_idx)["mg"].fourier_reg(uvect, xp.array(ac.support))
        uvect = alambda * uvect_ADA + rlambda * uvect + flambda * uvect_FDF
        return uvect

    W = LinearOperator(dtype=c_type, shape=(Mtot, Mtot), matvec=W_matvec)

    x0 = ac.estimate.reshape(-1)
    x0 = xp.array(x0)
    ADb = uregion.ADb.reshape(-1)
    ADb = xp.array(ADb)
    d = alambda * ADb + rlambda * x0
    callback = gprep.get_gprep(group_idx)["mg"].callback
    ret, info = cg(W, d, x0=x0, maxiter=maxiter, callback=callback)
    logger.log(f"WARNING: CG did not converge at rlambda = {rlambda}",level=1)

    ac_res = ret.reshape((M,) * 3)
    if not isinstance(ac_res, np.ndarray):
        ac_res = ac_res.get()
    if use_reciprocal_symmetry:
        assert np.all(np.isreal(ac_res))
    result.ac[:] = np.ascontiguousarray(ac_res.real)
    it_number = callback.counter
    logger.log(f"conf_index: {group_idx} recovered AC in {it_number} iterations.", level=1)
    image.show_volume(
        ac_res.real, settings.Mquat, f"autocorrelation_conf_{group_idx}_{generation}_{rank}.png"
    )
    #  if settings.debug_image:
    v1 = norm(ret)
    v2 = norm(W * ret - d)
    if not isinstance(v1, np.ndarray) and not isinstance(v1, float):
        v1 = v1.get()
        v2 = v2.get()

    summary.rank[0] = rank
    summary.rlambda[0] = rlambda
    summary.v1[0] = v1
    summary.v2[0] = v2


@task(leaf=True, privileges=[None, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def select_ac(generation, summary):
    iref = 0
    if generation == 0:
        # Heuristic: retain rank with highest lambda and high v1.
        idx = summary.v1 >= np.mean(summary.v1)
        imax = np.argmax(summary.rlambda[idx])
        iref = np.arange(len(summary.rank), dtype=np.int)[idx][imax]
    else:
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
    ac_result_subregion(results_p0, results_p[indx], point=group_idx)


@task(leaf=True, privileges=[RO("reciprocal")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def pixel_distance_rp_max_task(pixel_distance):
    return pixel_distance.reciprocal.max()


@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def solve_ac(
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
        str_mode=False,
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
        solve_ac_dict = prepare_solve_all_gens(slices_p, solve_ac_dict, str_mode)
        solve_ac_dict["slices_p"] = slices_p
        solve_ac_dict["ready_objs"] = ready_objs
        if orientations is None:
            orientations, orientations_p = get_random_orientations(N_images_per_rank)
        solve_ac_dict["orientations"] = orientations
        solve_ac_dict["orientations_p"] = orientations_p
    elif orientations is None:
        orientations, orientations_p = get_random_orientations(N_images_per_rank)
        solve_ac_dict = prepare_solve_all_gens(slices_p, solve_ac_dict, str_mode)
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

    get_nonuniform_positions(solve_ac_dict, N_procs, ready_objs, group_idx)

    ac = solve_ac_dict["ac"]
    if phased is None:
        pygion.fill(ac, "support", 1)
        pygion.fill(ac, "estimate", 0.0)
    else:
        phased_to_constrains(phased, ac)

    weights = 1
    prepare_solve(
        solve_ac_dict, slices_p, weights, M, Mtot, M_ups, N, use_reciprocal_symmetry, group_idx)

    results = solve_ac_dict["results"]
    results_p = solve_ac_dict["results_p"]
    results_r = solve_ac_dict["results_r"]

    alambda = 1
    rlambdas = Mtot / N * 2 ** (np.arange(N_procs) - N_procs / 2).astype(np.float)
    flambdas = 1e5 * 10 ** (np.arange(N_procs) - N_procs // 2).astype(np.float)
    uregion_p = solve_ac_dict["uregion_p"]
    uregion_ups_p = solve_ac_dict["uregion_ups_p"]
    summary_p = solve_ac_dict["summary_p"]
    summary = solve_ac_dict["summary"]
    reciprocal_extent = solve_ac_dict["reciprocal_extent"]
    for i in range(N_procs):
        solve(
            uregion_p[i],
            uregion_ups_p[i],
            ac,
            results_p[i],
            summary_p[i],
            conf_p[i],
            ready_objs[i],
            M,
            M_ups,
            Mtot,
            generation,
            i,
            alambda,
            rlambdas[i],
            flambdas[i],
            reciprocal_extent,
            use_reciprocal_symmetry,
            maxiter,
            group_idx,
            settings.N_conformations,
            point=i)

    iref = select_ac(generation, summary)
    # remove blocking call
    # return results_p[iref.get()], solve_ac_dict
    ac_result_task(results_r, results, results_p, iref, group_idx, point=group_idx)
    return results_r, solve_ac_dict


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
    logger = utils.Logger(True, settings)
    for i in range(settings.N_conformations):
        # check if converged
        if len(fsc) > 0 and check_convergence_task(fsc[i]).get():
            logger.log(f"conformation {i} HAS converged in solve_ac")
            assert create_regions is False
            results = solve_ac_dict[i]["results_r"]
            result_array.append(results)
        else:
            logger.log(f"conformation {i} has NOT converged in solve_ac")
            if str_mode:
                if orientations is not None:
                    results, solve_ac_dict[i] = solve_ac(solve_ac_dict[i],
                                                         generation, pixel_position,
                                                         pixel_distance,
                                                         slices_p, ready_objs, conf_p, i,
                                                         orientations[i], orientations_p[i],
                                                         phased[i], str_mode)
                    result_array.append(results)
                else:
                    results, solve_ac_dict[i] = solve_ac(solve_ac_dict[i],
                                                         generation, pixel_position,
                                                         pixel_distance,
                                                         slices_p, ready_objs, conf_p, i,
                                                         None, None,
                                                         None, str_mode)
                    result_array.append(results)
            else:
                if create_regions == False:
                    results, solve_ac_dict[i] = solve_ac(solve_ac_dict[i],
                                                         generation, pixel_position,
                                                         pixel_distance,
                                                         slices_p, ready_objs, conf_p, i,
                                                         orientations[i], orientations_p[i],
                                                         phased[i], str_mode)
                    result_array.append(results)
                else:
                    results, solve_ac_dict_entry = solve_ac(solve_ac_dict, generation, pixel_position,
                                                            pixel_distance,
                                                            slices_p, ready_objs, conf_p, i,
                                                            orientations, orientations_p, phased, str_mode)
                    solve_ac_array.append(solve_ac_dict_entry)
                    result_array.append(results)
    if create_regions:
        return result_array, solve_ac_array

    return result_array, solve_ac_dict
