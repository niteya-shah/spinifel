import matplotlib.pyplot as plt
import numpy             as np
import skopi             as skp
import PyNVTX            as nvtx
import pygion
import socket

from pygion import task, IndexLaunch, Partition, Region, RO, WD, Reduce, Tunable
from scipy.linalg        import norm
from scipy.ndimage       import gaussian_filter
from scipy.sparse.linalg import LinearOperator, cg

from spinifel import settings, autocorrelation, utils, image
from . import utils as lgutils


@task(leaf=True, privileges=[WD])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def gen_random_orientations(orientations, N_images_per_rank):
    orientations.quaternions[:] = skp.get_random_quat(N_images_per_rank)

@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def get_random_orientations(N_images_per_rank):
    if settings.use_single_prec:
        fields_dict = {"quaternions": pygion.float32}
    else:
        fields_dict = {"quaternions": pygion.float64}
    sec_shape = (4,)
    orientations, orientations_p = lgutils.create_distributed_region(
        N_images_per_rank, fields_dict, sec_shape)
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in range(N_procs):
        gen_random_orientations(orientations_p[i], N_images_per_rank, point=i)
    return orientations, orientations_p

@task(leaf=True, privileges=[RO, WD])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def gen_nonuniform_positions_v(nonuniform, nonuniform_v,
                               reciprocal_extent):
    reciprocal_extent = reciprocal_extent.get()
    mult = np.pi/(reciprocal_extent * settings.oversampling)
    nonuniform_v.H[:] = nonuniform.H.reshape(-1)*mult
    nonuniform_v.K[:] = nonuniform.K.reshape(-1)*mult
    nonuniform_v.L[:] = nonuniform.L.reshape(-1)*mult

@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def get_nonuniform_positions_v(nonuniform_p,
                               nonuniform_v_p,
                               reciprocal_extent,
                               N_procs):
    for i in range(N_procs):
        gen_nonuniform_positions_v(nonuniform_p[i],
                                   nonuniform_v_p[i],
                                   reciprocal_extent, point=i)

@task(leaf=True, privileges=[RO, WD, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def gen_nonuniform_positions(orientations, nonuniform, pixel_position):
     nonuniform.H[:], nonuniform.K[:], nonuniform.L[:] = autocorrelation.gen_nonuniform_positions(
         orientations.quaternions, pixel_position.reciprocal)

@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def get_nonuniform_positions(ac_dict, N_procs):
    orientations_p = ac_dict['orientations_p']
    nonuniform_p = ac_dict['nonuniform_p']
    pixel_position = ac_dict['pixel_position']
    for i in range(N_procs):
        gen_nonuniform_positions(
            orientations_p[i], nonuniform_p[i],
            pixel_position, point=i)

#equivalent to setup_linops
@task(privileges=[RO, WD, RO, RO, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def right_hand_ADb_task(slices, uregion, nonuniform_v,
                        ac, nuvect_p, M,
                        use_reciprocal_symmetry):
    if settings.verbosity > 0:
        print(f"{socket.gethostname()} started ADb.", flush=True)
    uregion.ADb[:] = autocorrelation.adjoint(
        nuvect_p.nuvect_Db,
        nonuniform_v.H,
        nonuniform_v.K,
        nonuniform_v.L,
        M,
        support=ac.support,
        use_recip_sym=use_reciprocal_symmetry)
    if settings.verbosity > 0:
        print(f"{socket.gethostname()} computed ADb.", flush=True)

@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def right_hand(slices_p, uregion_p, nonuniform_v_p,
               ac, nuvect_p, M, use_reciprocal_symmetry):
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()

    for i in range(N_procs):
        right_hand_ADb_task(slices_p[i], uregion_p[i],
                            nonuniform_v_p[i],
                            ac, nuvect_p[i], M,
                            use_reciprocal_symmetry, point=i)


@task(leaf=True, privileges=[WD, RO, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def prep_Fconv_task(uregion_ups, nonuniform_v, ac,
                    weights, M_ups, Mtot, N,
                    reciprocal_extent, use_reciprocal_symmetry):
    if settings.verbosity > 0:
        print(f"{socket.gethostname()} started Fconv.", flush=True)

    conv_ups = autocorrelation.adjoint(
        np.ones(N),
        nonuniform_v.H,
        nonuniform_v.K,
        nonuniform_v.L,
        M_ups,
        use_reciprocal_symmetry, support=None)
    uregion_ups.F_conv_[:] = np.fft.fftn(np.fft.ifftshift(conv_ups))
    if settings.verbosity > 0:
        print(f"{socket.gethostname()} computed Fconv.", flush=True)

@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def prep_Fconv(uregion_ups, nonuniform_v, nonuniform_v_p,
               ac, weights, M_ups, Mtot, N,
               reciprocal_extent, use_reciprocal_symmetry):
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    #for i in IndexLaunch([N_procs]):
    for i in range(N_procs):
        prep_Fconv_task(uregion_ups[i], nonuniform_v_p[i],
                        ac, weights, M_ups, Mtot, N,
                        reciprocal_extent, use_reciprocal_symmetry, point=i)


@task(leaf=True, privileges=[WD,RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def prep_nuvect(nuvect, slices, N, f_type):
    data = slices.data.reshape(-1)
    weights = np.ones(N, dtype=f_type)
    nuvect.nuvect_Db[:] = data * weights
    nuvect.nuvect[:] = np.ones_like(data)

#needed only once for all generations
@task(leaf=True, privileges=[WD("F_antisupport")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def prep_Fantisupport(uregion, M):
    lu = np.linspace(-np.pi, np.pi, M)
    Hu_, Ku_, Lu_ = np.meshgrid(lu, lu, lu, indexing='ij')
    Qu_ = np.around(np.sqrt(Hu_**2 + Ku_**2 + Lu_**2), 4)
    uregion.F_antisupport[:] = Qu_ > np.pi / settings.oversampling
    # Generate an antisupport in Fourier space, which has zeros in the central
    # sphere and ones in the high-resolution corners.
    Fantisup = uregion.F_antisupport
    assert np.all(Fantisup[:] == Fantisup[::-1, :, :])
    assert np.all(Fantisup[:] == Fantisup[:, ::-1, :])
    assert np.all(Fantisup[:] == Fantisup[:, :, ::-1])
    assert np.all(Fantisup[:] == Fantisup[::-1, ::-1, ::-1])


# create all the region
# initialize regions
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def prepare_solve_all_gens(slices_p):

    solve_dict = {}
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    N_images_per_rank = slices_p[0].ispace.domain.extent[0]
    M = settings.M

    cmpx_type = None
    np_complx_type = None
    if settings.use_single_prec:
        cmpx_type = pygion.complex64
        np_complx_type = np.complex64,
        float_type = pygion.float32
    else:
        cmpx_type = pygion.complex128
        np_complx_type = np.complex128,
        float_type = pygion.float64

    fields_dict = {"nuvect_Db": cmpx_type,
                   "nuvect": cmpx_type}
    N_vals_per_rank = N_images_per_rank * utils.prod(settings.reduced_det_shape)
    sec_shape = ()
    nuvect, nuvect_p = lgutils.create_distributed_region(
        N_vals_per_rank, fields_dict, sec_shape)
    solve_dict['nuvect'] = nuvect
    solve_dict['nuvect_p'] = nuvect_p

    fields_dict = {"ADb": float_type, "F_antisupport": pygion.bool_}
    uregion, uregion_p = lgutils.create_distributed_region(
        M, fields_dict, (M,M,))
    solve_dict['uregion'] = uregion
    solve_dict['uregion_p'] = uregion_p

    # For upsampled convolution technique
    M_ups = settings.M_ups
    fields_dict =  {"F_conv_": cmpx_type}
    uregion_ups, uregion_ups_p = lgutils.create_distributed_region(
        M_ups, fields_dict, (M_ups,M_ups,))
    solve_dict['uregion_ups'] = uregion_ups
    solve_dict['uregion_ups_p'] = uregion_ups_p

    # H, K, L
    fields_dict = {"H": pygion.float64, "K": pygion.float64,
                   "L": pygion.float64}
    # nonuniform_v
    nonuniform_v, nonuniform_v_p = lgutils.create_distributed_region(
        N_vals_per_rank, fields_dict, sec_shape)
    solve_dict['nonuniform_v'] = nonuniform_v
    solve_dict['nonuniform_v_p'] = nonuniform_v_p

    # nonuniform
    fields_dict = {"H": float_type, "K": float_type,
                   "L": float_type}
    sec_shape = settings.reduced_det_shape
    nonuniform, nonuniform_p = lgutils.create_distributed_region(
        N_images_per_rank, fields_dict, sec_shape)
    solve_dict['nonuniform'] = nonuniform
    solve_dict['nonuniform_p'] = nonuniform_p

    # ac
    ac = Region((M,)*3,
                {"support": pygion.float64,
                 "estimate": pygion.float64})
    solve_dict['ac'] = ac

    # summary
    summary = Region((N_procs,),
                     {"rank": pygion.int32, "rlambda": pygion.float64, "v1": pygion.float64, "v2": pygion.float64})
    summary_p = Partition.equal(summary, (N_procs,))

    solve_dict['summary'] = summary
    solve_dict['summary_p'] = summary_p

    results = Region((N_procs * M, M, M), {"ac": pygion.float64})
    results_p = Partition.restrict(results, (N_procs,), [[M], [0], [0]], [M, M, M])
    results_r = Region((M, M, M), {"ac": pygion.float64})

    solve_dict['results'] = results
    solve_dict['results_p'] = results_p
    solve_dict['results_r'] = results_r

    if settings.use_single_prec:
        f_type = np.float32
    else:
        f_type = np.float64

    # compute Fantisupport
    for i in range(N_procs):
        prep_Fantisupport(uregion_p[i], M, point=i)
        prep_nuvect(nuvect_p[i], slices_p[i], N_vals_per_rank,
                    f_type, point=i)

    # create a dictionary of regions/partitions
    return solve_dict

@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def prepare_solve(solve_ac_dict, slices_p,
                  weights, M, Mtot, M_ups, N,
                  use_reciprocal_symmetry):

    N_images_per_rank = slices_p[0].ispace.domain.extent[0]
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    ac = solve_ac_dict['ac']
    nonuniform =  solve_ac_dict['nonuniform']
    nonuniform_p =  solve_ac_dict['nonuniform_p']
    nonuniform_v = solve_ac_dict['nonuniform_v']
    nonuniform_v_p = solve_ac_dict['nonuniform_v_p']
    reciprocal_extent = solve_ac_dict['reciprocal_extent']
    uregion_ups_p = solve_ac_dict['uregion_ups_p']
    nuvect_p = solve_ac_dict['nuvect_p']
    uregion_p = solve_ac_dict['uregion_p']
    get_nonuniform_positions_v(nonuniform_p, nonuniform_v_p,
                               reciprocal_extent, N_procs)

    prep_Fconv(uregion_ups_p, nonuniform_v, nonuniform_v_p,
               ac, weights, M_ups, Mtot, N,
               reciprocal_extent, use_reciprocal_symmetry)

    right_hand(slices_p, uregion_p,
               nonuniform_v_p,
               ac, nuvect_p, M,
               use_reciprocal_symmetry)

@task(leaf=True, privileges=[RO("ac"), WD("support", "estimate")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def phased_to_constrains(phased, ac):
    ac_smoothed = gaussian_filter(phased.ac, 0.5)
    ac.support[:] = (ac_smoothed > 1e-12)
    ac.estimate[:] = phased.ac * ac.support

@task(leaf=True, privileges=[RO, RO, RO, WD, WD])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def solve(uregion, uregion_ups, ac, result, summary,
          M, M_ups, Mtot,
          generation, rank, alambda, rlambda, flambda,
          reciprocal_extent, use_reciprocal_symmetry, maxiter):
    if settings.verbosity > 0:
        print(f" Rank {rank} started solve", flush=True)
    """Solve the W @ x = d problem.

    W = al*A_adj*Da*A + rl*I  + fl*F_adj*Df*F
    d = al*A_adj*Da*b + rl*x0 + 0

    Where:
        A represents the NUFFT operator
        A_adj its adjoint
        I the identity
        F the FFT operator
        F_adj its atjoint
        Da, Df weights
        b the data
        x0 the initial guess (ac_estimate)
    """
    def W_matvec(uvect):

        """Define W part of the W @ x = d problem."""
        assert use_reciprocal_symmetry, "Complex AC are not supported."
        assert np.all(np.isreal(uvect))

        uvect_ADA = autocorrelation.core_problem_convolution(
            uvect, M, uregion_ups.F_conv_, M_ups, ac.support, use_reciprocal_symmetry)
        uvect_FDF = autocorrelation.fourier_reg(
            uvect, ac.support, uregion.F_antisupport, M, use_reciprocal_symmetry)
        uvect = alambda*uvect_ADA + rlambda*uvect + flambda*uvect_FDF
        return uvect

    np_complx_type = np.complex128
    if settings.use_single_prec:
        np_complx_type = np.complex64

    W = LinearOperator(
        dtype=np_complx_type,
        shape=(Mtot, Mtot),
        matvec=W_matvec)

    x0 = ac.estimate.reshape(-1)
    ADb = uregion.ADb.reshape(-1)
    d = alambda*ADb + rlambda*x0
    def callback(xk):
        callback.counter += 1
    callback.counter = 0

    ret, info = cg(W, d, x0=x0, maxiter=maxiter, callback=callback)
    ac_res = ret.reshape((M,)*3)
    if use_reciprocal_symmetry:
        assert np.all(np.isreal(ac_res))
    result.ac[:] = np.ascontiguousarray(ac_res.real)
    it_number = callback.counter
    if settings.verbosity > 0:
        print(f"Rank {rank} recovered AC in {it_number} iterations.", flush=True)
    image.show_volume(ac_res.real, settings.Mquat,
                      f"autocorrelation_{generation}_{rank}.png")
    v1 = norm(ret)
    v2 = norm(W*ret-d)
    summary.rank[0] = rank
    summary.rlambda[0] = rlambda
    summary.v1[0] = v1
    summary.v2[0] = v2


@task(leaf=True, privileges=[None, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def select_ac(generation, summary):
    iref=0
    if generation == 0:
        # Heuristic: retain rank with highest lambda and high v1.
        idx = summary.v1 >= np.mean(summary.v1)
        imax = np.argmax(summary.rlambda[idx])
        iref = np.arange(len(summary.rank), dtype=np.int)[idx][imax]
    else:
        # Take corner of L-curve: min (v1+v2)
        iref = np.argmin(summary.v1+summary.v2)
    return iref

@task(leaf=True, privileges=[WD, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def ac_result_subregion(results_p0, results):
    results_p0.ac[:] = results.ac[:]

#partition the region and use only the iref.get() subregion
@task(inner=True, privileges=[WD, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def ac_result_task(results_p0, results, results_p, iref):
    indx = iref.get()
    ac_result_subregion(results_p0, results_p[indx])

@task(leaf=True, privileges=[RO("reciprocal")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def  pixel_distance_rp_max_task(pixel_distance):
    return pixel_distance.reciprocal.max()

@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def solve_ac(solve_ac_dict,
             generation,
             pixel_position,
             pixel_distance,
             slices_p,
             orientations=None,
             orientations_p=None,
             phased=None):

    M = settings.M
    M_ups = settings.M_ups  # For upsampled convolution technique
    Mtot = M**3
    N_images_per_rank = slices_p[0].ispace.domain.extent[0]
    N = N_images_per_rank * utils.prod(settings.reduced_det_shape)
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    use_reciprocal_symmetry = True
    maxiter = settings.solve_ac_maxiter
    fill_orientations = False

    if orientations is None:
        fill_orientations = True
        orientations, orientations_p = get_random_orientations(N_images_per_rank)
        solve_ac_dict = prepare_solve_all_gens(slices_p)
        solve_ac_dict['reciprocal_extent'] = pixel_distance_rp_max_task(pixel_distance)
        solve_ac_dict['orientations'] = orientations
        solve_ac_dict['orientations_p'] = orientations_p
        solve_ac_dict['pixel_position'] = pixel_position
        solve_ac_dict['pixel_distance'] = pixel_distance
        solve_ac_dict['slices_p'] = slices_p

    get_nonuniform_positions(solve_ac_dict, N_procs)

    # orientations region can be garbage collected
    if fill_orientations:
        pygion.fill(orientations, "quaternions", 0.)
    ac = solve_ac_dict['ac']
    if phased is None:
        pygion.fill(ac, "support", 1.)
        pygion.fill(ac, "estimate", 0.)
    else:
        phased_to_constrains(phased, ac)

    weights = 1
    prepare_solve(solve_ac_dict, slices_p,
                  weights, M, Mtot, M_ups, N,
                  use_reciprocal_symmetry)

    results = solve_ac_dict['results']
    results_p = solve_ac_dict['results_p']
    results_r = solve_ac_dict['results_r']

    alambda = 1
    rlambdas = Mtot/N * 2**(np.arange(N_procs) - N_procs/2).astype(np.float)
    flambdas = 1e5 * 10**(np.arange(N_procs) - N_procs//2).astype(np.float)
    uregion_p = solve_ac_dict['uregion_p']
    uregion_ups_p = solve_ac_dict['uregion_ups_p']
    summary_p =  solve_ac_dict['summary_p']
    summary = solve_ac_dict['summary']
    reciprocal_extent = solve_ac_dict['reciprocal_extent']
    for i in range(N_procs):
        solve(
            uregion_p[i], uregion_ups_p[i], ac, results_p[i], summary_p[i],
            M, M_ups, Mtot,
            generation, i, alambda, rlambdas[i], flambdas[i],
            reciprocal_extent, use_reciprocal_symmetry, maxiter, point=i)

    iref = select_ac(generation, summary)
    # At this point, I just want to chose one of the results as reference.
    # I tried to have `results` as a partition and copy into a region,
    # but I couldn't get it to work.
    # ac is in results_p[iref]
    #remove blocking call
    ac_result_task(results_r, results, results_p, iref)
    return results_r, solve_ac_dict
    #return results_p[iref.get()], solve_ac_dict


