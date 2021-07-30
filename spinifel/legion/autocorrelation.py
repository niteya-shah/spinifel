import matplotlib.pyplot as plt
import numpy.matlib
import numpy             as np
import skopi             as skp
import PyNVTX            as nvtx
import pygion
import socket

from pygion import task, IndexLaunch, Partition, Region, RO, WD, Reduce, Tunable
from scipy.linalg        import norm
from scipy.ndimage       import gaussian_filter
from scipy.sparse.linalg import LinearOperator, cg

from spinifel import parms, autocorrelation, utils, image
from . import utils as lgutils



@task(privileges=[WD])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def gen_random_orientations(orientations, N_images_per_rank):
    orientations.quaternions[:] = skp.get_random_quat(N_images_per_rank)



@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def get_random_orientations():
    N_ranks_per_node = parms.N_ranks_per_node
    N_images_per_rank = parms.N_images_per_rank
    fields_dict = {"quaternions": pygion.float64}
    sec_shape = (4,)
    orientations, orientations_p = lgutils.create_distributed_region_per_node(
        N_images_per_rank, fields_dict, sec_shape)
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    N_nodes = N_procs // N_ranks_per_node
    for i in IndexLaunch([N_nodes]):
        gen_random_orientations(orientations_p[i], N_images_per_rank)
    return orientations, orientations_p



@task(privileges=[RO, WD])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def gen_nonuniform_positions_v(nonuniform, nonuniform_v, reciprocal_extent):
    nonuniform_v.H[:] = (nonuniform.H.flatten()
        / reciprocal_extent * np.pi / parms.oversampling)
    nonuniform_v.K[:] = (nonuniform.K.flatten()
        / reciprocal_extent * np.pi / parms.oversampling)
    nonuniform_v.L[:] = (nonuniform.L.flatten()
        / reciprocal_extent * np.pi / parms.oversampling)



@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def create_nonuniform_positions_v(nonuniform_p, reciprocal_extent):
    """Flatten and calibrate nonuniform positions."""
    N_vals_per_rank = (
        parms.N_images_per_rank * utils.prod(parms.reduced_det_shape))
    fields_dict = {"H": pygion.float64, "K": pygion.float64,
                   "L": pygion.float64}
    sec_shape = ()
    nonuniform_v, nonuniform_v_p = lgutils.create_distributed_region(
        N_vals_per_rank, fields_dict, sec_shape)
    return nonuniform_v, nonuniform_v_p



@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def fill_nonuniform_positions_v(nonuniform_p, reciprocal_extent, nonuniform_v_p):
    """Flatten and calibrate nonuniform positions."""
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in IndexLaunch([N_procs]):
        gen_nonuniform_positions_v(nonuniform_p[i], nonuniform_v_p[i],
                                   reciprocal_extent)



@task(privileges=[RO, WD, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def gen_nonuniform_positions(orientations, nonuniform, pixel_position):
    H, K, L = autocorrelation.gen_nonuniform_positions(
        orientations.quaternions, pixel_position.reciprocal)
    nonuniform.H[:] = H
    nonuniform.K[:] = K
    nonuniform.L[:] = L



@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def create_nonuniform_positions():
    N_images_per_rank = parms.N_images_per_rank
    fields_dict = {"H": pygion.float64, "K": pygion.float64,
                   "L": pygion.float64}
    sec_shape = parms.reduced_det_shape
    nonuniform, nonuniform_p = lgutils.create_distributed_region(
        N_images_per_rank, fields_dict, sec_shape)
    return nonuniform, nonuniform_p



@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def fill_nonuniform_positions(orientations_p, pixel_position, nonuniform_p):
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in IndexLaunch([N_procs]):
        gen_nonuniform_positions(
            orientations_p[i], nonuniform_p[i], pixel_position)



@task(privileges=[RO, Reduce('+', 'ADb'), RO, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def right_hand_ADb_task(slices, uregion, nonuniform_v, ac, weights, M, N,
                        reciprocal_extent, use_reciprocal_symmetry):
    if parms.verbosity > 0:
        print(f"{socket.gethostname()} started ADb.", flush=True)
    N_images_per_rank = parms.N_images_per_rank
    N_pixels_per_image = N / N_images_per_rank
    data = (slices.data * (M**3/N_pixels_per_image)).flatten()
    nuvect_Db = data * weights
    uregion.ADb[:] += autocorrelation.adjoint(
        nuvect_Db,
        nonuniform_v.H,
        nonuniform_v.K,
        nonuniform_v.L,
        ac.support, M,
        reciprocal_extent, use_reciprocal_symmetry
    )
    if parms.verbosity > 0:
        print(f"{socket.gethostname()} computed ADb.", flush=True)



@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def right_hand(slices_p, uregion, nonuniform_v_p,
               ac, weights, M, N,
               reciprocal_extent, use_reciprocal_symmetry):
    pygion.fill(uregion, "ADb", 0.)
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in IndexLaunch([N_procs]):
        right_hand_ADb_task(slices_p[i], uregion, nonuniform_v_p[i],
                            ac, weights, M, N,
                            reciprocal_extent, use_reciprocal_symmetry)



@task(privileges=[RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def Db_squared_task(slices, weights):
    if parms.verbosity > 0:
        print(f"{socket.gethostname()} started Dbsquared.", flush=True)
    Db_squared = np.sum(norm(slices.data.reshape(slices.data.shape[0],-1))**2, axis=-1)
    return Db_squared



@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def get_Db_squared(slices_p, weights):
    futures = []
    Db_squared = 0
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in IndexLaunch([N_procs]):
        futures.append(Db_squared_task(slices_p[i], weights))
    for i, future in enumerate(futures):
        print('got %s' % future.get())
        Db_squared += future.get()
    return Db_squared



@task(privileges=[Reduce('+', 'F_conv_'), RO, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def prep_Fconv_task(uregion_ups, nonuniform_v, ac, weights, M_ups, M, N,
                    reciprocal_extent, use_reciprocal_symmetry):
    if parms.verbosity > 0:
        print(f"{socket.gethostname()} started Fconv.", flush=True)
    conv_ups = autocorrelation.adjoint(
        np.ones(N),
        nonuniform_v.H,
        nonuniform_v.K,
        nonuniform_v.L,
        1, M_ups,
        reciprocal_extent, use_reciprocal_symmetry
    )
    uregion_ups.F_conv_[:] += np.fft.fftn(np.fft.ifftshift(conv_ups)) / M**3
    if parms.verbosity > 0:
        print(f"{socket.gethostname()} computed Fconv.", flush=True)



@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def prep_Fconv(uregion_ups, nonuniform_v_p,
               ac, weights, M_ups, M, N,
               reciprocal_extent, use_reciprocal_symmetry):
    pygion.fill(uregion_ups, "F_conv_", 0.)
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in IndexLaunch([N_procs]):
        prep_Fconv_task(uregion_ups, nonuniform_v_p[i],
                        ac, weights, M_ups, M, N,
                        reciprocal_extent, use_reciprocal_symmetry)



@task(privileges=[WD("F_antisupport")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def prep_Fantisupport(uregion, M):
    lu = np.linspace(-np.pi, np.pi, M)
    Hu_, Ku_, Lu_ = np.meshgrid(lu, lu, lu, indexing='ij')
    Qu_ = np.sqrt(Hu_**2 + Ku_**2 + Lu_**2)
    uregion.F_antisupport[:] = Qu_ > np.pi / parms.oversampling

    Fantisup = uregion.F_antisupport
    assert np.all(Fantisup[:] == Fantisup[::-1, :, :])
    assert np.all(Fantisup[:] == Fantisup[:, ::-1, :])
    assert np.all(Fantisup[:] == Fantisup[:, :, ::-1])
    assert np.all(Fantisup[:] == Fantisup[::-1, ::-1, ::-1])



@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def setup_prepare_solve(nonuniform, nonuniform_p,
                  M, M_ups,
                  reciprocal_extent):
    nonuniform_v, nonuniform_v_p = create_nonuniform_positions_v(
        nonuniform_p, reciprocal_extent)
    uregion = Region((M,)*3,
                     {"ADb": pygion.float64, "F_antisupport": pygion.float64})
    uregion_ups = Region((M_ups,)*3, {"F_conv_": pygion.complex128})
    return nonuniform_v, nonuniform_v_p, uregion, uregion_ups



@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def prepare_solve(slices_p, nonuniform_p, nonuniform_v_p, uregion, uregion_ups,
                  ac, weights, M, M_ups, N,
                  reciprocal_extent, use_reciprocal_symmetry):
    fill_nonuniform_positions_v(
        nonuniform_p, reciprocal_extent, nonuniform_v_p)
    prep_Fconv(uregion_ups, nonuniform_v_p,
               ac, weights, M_ups, M, N,
               reciprocal_extent, use_reciprocal_symmetry)
    right_hand(slices_p, uregion, nonuniform_v_p,
               ac, weights, M, N,
               reciprocal_extent, use_reciprocal_symmetry)
    prep_Fantisupport(uregion, M)



@task(privileges=[RO("ac"), WD("support", "estimate")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def phased_to_constrains(phased, ac):
    ac_smoothed = gaussian_filter(phased.ac, 0.5)
    ac.support[:] = (ac_smoothed > 1e-12).astype(np.float64)
    ac.estimate[:] = phased.ac * ac.support



@task(privileges=[RO, RO, RO, WD, WD])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def solve(uregion, uregion_ups, ac, result, solve_summary,
          weights, M, M_ups, N,
          generation, rank, rlambda, flambda, Db_squared,
          reciprocal_extent, use_reciprocal_symmetry, maxiter):
    """Solve the W @ x = d problem.

    W = A_adj*Da*A + rl*I  + fl*F_adj*Df*F
    d = A_adj*Da*b + rl*x0 + 0

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
        uvect = uvect_ADA + rlambda*uvect + flambda*uvect_FDF
        return uvect

    def W0_matvec(uvect):
        """Define W part of the W @ x = d problem."""
        assert use_reciprocal_symmetry, "Complex AC are not supported."
        assert np.all(np.isreal(uvect))

        uvect_ADA = autocorrelation.core_problem_convolution(
            uvect, M, uregion_ups.F_conv_, M_ups, ac.support, use_reciprocal_symmetry)
        uvect = uvect_ADA 
        return uvect

    W = LinearOperator(
        dtype=np.complex128,
        shape=(M**3, M**3),
        matvec=W_matvec)

    W0 = LinearOperator(
        dtype=np.complex128,
        shape=(M**3, M**3),
        matvec=W0_matvec)

    x0 = ac.estimate.astype(np.float64).flatten()
    ADb = uregion.ADb.flatten()
    d = ADb + rlambda*x0
    d0 = ADb

    # Log central slice L~=0
    if comm.rank == (2 if parms.use_psana else 0):
        idx = np.abs(L) < reciprocal_extent * .01
        plt.scatter(H[idx], K[idx], c=slices_[idx], s=1, norm=LogNorm())
        plt.axis('equal')
        plt.colorbar()
        plt.savefig(parms.out_dir / f"star_{generation}.png")
        plt.cla()
        plt.clf()

    def callback(xk):
        callback.counter += 1
    callback.counter = 0

    ret, info = cg(W, d, x0=x0, maxiter=maxiter, callback=callback)
    print('info =', info)
    if info != 0:
        print(f'WARNING: CG did not converge at rlambda = {rlambda}')

    ret /= M**3
    d0 /= M**3
    soln = norm(ret)**2
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    Ntot = N * N_procs # total number of pixels
    resid = (np.dot(ret,W0.matvec(ret)-2*d0) + Db_squared) / Ntot # residual norm

    ac_res = (ret*M**3).reshape((M,)*3)
    if use_reciprocal_symmetry:
        assert np.all(np.isreal(ac_res))
    result.ac[:] = ac_res.real
    it_number = callback.counter

    if parms.verbosity > 0:
        print(f"{socket.gethostname()} - ", end='')
    print(f"Rank {rank} recovered AC in {it_number} iterations.", flush=True)
    image.show_volume(result.ac[:], parms.Mquat,
                      f"autocorrelation_{generation}_{rank}.png")

    solve_summary.rank[0] = rank
    solve_summary.rlambda[0] = rlambda
    solve_summary.info[0] = info
    solve_summary.soln[0] = soln
    solve_summary.resid[0] = resid



@task(privileges=[None, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def select_ac(generation, solve_summary):
    print('len(solve_summary.rank) =', len(solve_summary.rank))
    print('solve_summary.rank =', solve_summary.rank)
    print('solve_summary.rlambda =', solve_summary.rlambda)
    print('solve_summary.info =', solve_summary.info)
    print('solve_summary.soln =', solve_summary.soln)
    print('solve_summary.resid =', solve_summary.resid) 

    converged_idx = [i for i, info in enumerate(solve_summary.info) if info == 0]
    ranks = np.array(solve_summary.rank)[converged_idx]
    lambdas = np.array(solve_summary.rlambda)[converged_idx]
    solns = np.array(solve_summary.soln)[converged_idx]
    resids = np.array(solve_summary.resid)[converged_idx]    
    
    # Take corner of L-curve
    valuePair = np.array([resids, solns]).T
    valuePair = np.array(sorted(valuePair , key=lambda k: [k[0], k[1]])) # sort the corner candidates in increasing order
    lambdas = np.sort(lambdas) # sort lambdas in increasing order
    allCoord = np.log(valuePair) # coordinates of the loglog L-curve
    nPoints = len(resids)
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * numpy.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    iref = np.argmax(distToLine)
    print('iref =', iref)
    ref_rank = ranks[iref]

    # Log L-curve plot
    fig, axes = plt.subplots(figsize=(10.0, 24.0), nrows=3, ncols=1)
    axes[0].loglog(lambdas, valuePair.T[1])
    axes[0].loglog(lambdas[iref], valuePair[iref][1], "rD")
    axes[0].set_xlabel("$\lambda$")
    axes[0].set_ylabel("Solution norm $||x||_{2}$")
    axes[1].loglog(lambdas, valuePair.T[0])
    axes[1].loglog(lambdas[iref], valuePair[iref][0], "rD")
    axes[1].set_xlabel("$\lambda$")
    axes[1].set_ylabel("Residual norm $||Ax-b||_{2}$")
    axes[2].loglog(valuePair.T[0], valuePair.T[1]) # L-curve
    axes[2].loglog(valuePair[iref][0], valuePair[iref][1], "rD")
    axes[2].set_xlabel("Residual norm $||Ax-b||_{2}$")
    axes[2].set_ylabel("Solution norm $||x||_{2}$")
    fig.tight_layout()
    plt.savefig(parms.out_dir / f"solve_summary_{generation}.png")
    plt.close('all')

    print(f"Keeping result from rank {ref_rank}.", flush=True)

    return iref

@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def setup_solve_ac(pixel_position,
                   pixel_distance):
    print("setup_solve_ac")
    M = parms.M
    M_ups = parms.M_ups  # For upsampled convolution technique
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    print("N_procs =", N_procs)
    reciprocal_extent = pixel_distance.reciprocal.max()

    orientations, orientations_p = get_random_orientations()
    nonuniform, nonuniform_p = create_nonuniform_positions()

    ac = Region((M,)*3,
                {"support": pygion.float64, "estimate": pygion.float64})

    nonuniform_v, nonuniform_v_p, uregion, uregion_ups = setup_prepare_solve(
        nonuniform, nonuniform_p, M, M_ups, reciprocal_extent)

    results = Region((N_procs * M, M, M), {"ac": pygion.float64})
    results_p = Partition.restrict(results, (N_procs,), [[M], [0], [0]], [M, M, M])

    solve_summary = Region((N_procs,),
                {"rank": pygion.int32, "rlambda": pygion.float64, "info": pygion.int32, "soln": pygion.float64, "resid": pygion.float64})
    solve_summary_p = Partition.equal(solve_summary, (N_procs,))

    return (orientations, orientations_p,
            nonuniform, nonuniform_p,
            nonuniform_v, nonuniform_v_p,
            ac, uregion, uregion_ups,
            results, results_p,
            solve_summary, solve_summary_p)

@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def solve_ac(generation,
             pixel_position,
             pixel_distance,
             slices, slices_p,
             orientations_p,
             nonuniform_p,
             nonuniform_v_p,
             ac, uregion, uregion_ups,
             results_p,
             solve_summary, solve_summary_p,
             phased=None):
    M = parms.M
    M_ups = parms.M_ups  # For upsampled convolution technique
    N_images_per_rank = parms.N_images_per_rank # N images per rank
    N = N_images_per_rank * utils.prod(parms.reduced_det_shape) # N images per rank x number of pixels per image = number of pixels per rank
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    reciprocal_extent = pixel_distance.reciprocal.max()
    use_reciprocal_symmetry = True
    maxiter = parms.solve_ac_maxiter

    fill_nonuniform_positions(
        orientations_p, pixel_position, nonuniform_p)

    if phased is None:
        pygion.fill(ac, "support", 1.)
        pygion.fill(ac, "estimate", 0.)
    else:
        phased_to_constrains(phased, ac)
    weights = 1

    prepare_solve(
        slices_p, nonuniform_p, nonuniform_v_p, uregion, uregion_ups,
        ac, weights, M, M_ups, N,
        reciprocal_extent, use_reciprocal_symmetry)

    Db_squared = get_Db_squared(slices_p, weights)

    #rlambdas = 100**(np.arange(N_procs) - N_procs/2)
    rlambdas = np.logspace(-8, 8, N_procs)
    flambda = 0

    for i in IndexLaunch((N_procs,)):
        solve(
            uregion, uregion_ups, ac, results_p[i], solve_summary_p[i],
            weights, M, M_ups, N,
            generation, i, rlambdas[i], flambda, Db_squared,
            reciprocal_extent, use_reciprocal_symmetry, maxiter)

    iref = select_ac(generation, solve_summary)
    # At this point, I just want to chose one of the results as reference.
    # I tried to have `results` as a partition and copy into a region,
    # but I couldn't get it to work.
    return results_p[iref.get()]
