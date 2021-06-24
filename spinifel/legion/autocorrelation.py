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
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def gen_random_orientations(orientations, N_images_per_rank):
    orientations.quaternions[:] = skp.get_random_quat(N_images_per_rank)



@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def get_random_orientations():
    N_images_per_rank = parms.N_images_per_rank
    fields_dict = {"quaternions": pygion.float32}
    sec_shape = (4,)
    orientations, orientations_p = lgutils.create_distributed_region(
        N_images_per_rank, fields_dict, sec_shape)
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in IndexLaunch([N_procs]):
        gen_random_orientations(orientations_p[i], N_images_per_rank)
    return orientations, orientations_p



@task(privileges=[RO, WD])
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def gen_nonuniform_positions_v(nonuniform, nonuniform_v, reciprocal_extent):
    nonuniform_v.H[:] = (nonuniform.H.flatten()
        / reciprocal_extent * np.pi / parms.oversampling)
    nonuniform_v.K[:] = (nonuniform.K.flatten()
        / reciprocal_extent * np.pi / parms.oversampling)
    nonuniform_v.L[:] = (nonuniform.L.flatten()
        / reciprocal_extent * np.pi / parms.oversampling)



@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def get_nonuniform_positions_v(nonuniform, nonuniform_p, reciprocal_extent):
    """Flatten and calibrate nonuniform positions."""
    N_vals_per_rank = (
        parms.N_images_per_rank * utils.prod(parms.reduced_det_shape))
    fields_dict = {"H": pygion.float64, "K": pygion.float64,
                   "L": pygion.float64}
    sec_shape = ()
    nonuniform_v, nonuniform_v_p = lgutils.create_distributed_region(
        N_vals_per_rank, fields_dict, sec_shape)
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in IndexLaunch([N_procs]):
        gen_nonuniform_positions_v(nonuniform_p[i], nonuniform_v_p[i],
                                   reciprocal_extent)
    return nonuniform_v, nonuniform_v_p



@task(privileges=[RO, WD, RO])
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def gen_nonuniform_positions(orientations, nonuniform, pixel_position):
    H, K, L = autocorrelation.gen_nonuniform_positions(
        orientations.quaternions, pixel_position.reciprocal)
    nonuniform.H[:] = H
    nonuniform.K[:] = K
    nonuniform.L[:] = L



@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def get_nonuniform_positions(orientations, orientations_p, pixel_position):
    N_images_per_rank = parms.N_images_per_rank
    fields_dict = {"H": pygion.float32, "K": pygion.float32,
                   "L": pygion.float32}
    sec_shape = parms.reduced_det_shape
    nonuniform, nonuniform_p = lgutils.create_distributed_region(
        N_images_per_rank, fields_dict, sec_shape)
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in IndexLaunch([N_procs]):
        gen_nonuniform_positions(
            orientations_p[i], nonuniform_p[i], pixel_position)
    return nonuniform, nonuniform_p



@task(privileges=[RO, Reduce('+', 'ADb'), RO, RO])
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def right_hand_ADb_task(slices, uregion, nonuniform_v, ac, weights, M, N,
                        reciprocal_extent, use_reciprocal_symmetry):
    if parms.verbosity > 0:
        print(f"{socket.gethostname()} started ADb.", flush=True)
    N_images_per_rank = parms.N_images_per_rank
    N_pixels_per_image = N / N_images_per_rank
    print('N_pixels_per_image in right_hand =', N_pixels_per_image)
    data = (slices.data * (M**3/N_pixels_per_image)).flatten()
    nuvect_Db = data * weights
    print('nuvect_Db =', nuvect_Db)
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
def right_hand(slices, slices_p, uregion, nonuniform_v, nonuniform_v_p,
               ac, weights, M, N,
               reciprocal_extent, use_reciprocal_symmetry):
    pygion.fill(uregion, "ADb", 0.)
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in IndexLaunch([N_procs]):
        right_hand_ADb_task(slices_p[i], uregion, nonuniform_v_p[i],
                            ac, weights, M, N,
                            reciprocal_extent, use_reciprocal_symmetry)



@task(privileges=[RO])
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def Db_squared_task(slices, weights, M, N):
    if parms.verbosity > 0:
        print(f"{socket.gethostname()} started Dbsquared.", flush=True)
    N_images_per_rank = parms.N_images_per_rank
    N_pixels_per_image = N / N_images_per_rank
    print('weights =', weights)
    print('N =', N)
    print('N_images_per_rank =', N_images_per_rank)
    print('N_pixels_per_image =', N_pixels_per_image)
    Db_squared = np.sum(norm(slices.data.reshape(slices.data.shape[0],-1) * weights * (M**3/N_pixels_per_image))**2, axis=-1)
    print("Db_squared =", Db_squared)
    return Db_squared



@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def get_Db_squared(slices, slices_p, weights, M, N):
    futures = []
    Db_squared = 0
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in IndexLaunch([N_procs]):
        futures.append(Db_squared_task(slices_p[i], weights, M, N))
    for i, future in enumerate(futures):
        print('got %s' % future.get())
        Db_squared += future.get()
    print("sum of Db_squared =", Db_squared)
    return Db_squared



@task(privileges=[Reduce('+', 'F_conv_'), RO, RO])
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
    #uregion_ups.F_conv_[:] += np.fft.fftn(np.fft.ifftshift(conv_ups)) / M**3
    uregion_ups.F_conv_[:] += conv_ups
    if parms.verbosity > 0:
        print(f"{socket.gethostname()} computed Fconv.", flush=True)



@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def prep_Fconv(uregion_ups, nonuniform_v, nonuniform_v_p,
               ac, weights, M_ups, M, N,
               reciprocal_extent, use_reciprocal_symmetry):
    pygion.fill(uregion_ups, "F_conv_", 0.)
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in IndexLaunch([N_procs]):
        prep_Fconv_task(uregion_ups, nonuniform_v_p[i],
                        ac, weights, M_ups, M, N,
                        reciprocal_extent, use_reciprocal_symmetry)
    uregion_ups.F_conv_[:] = np.fft.fftn(np.fft.ifftshift(uregion_ups.F_conv_)) / M**3



@task(privileges=[WD("F_antisupport")])
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
def prepare_solve(slices, slices_p, nonuniform, nonuniform_p,
                  ac, weights, M, M_ups, N,
                  reciprocal_extent, use_reciprocal_symmetry):
    nonuniform_v, nonuniform_v_p = get_nonuniform_positions_v(
        nonuniform, nonuniform_p, reciprocal_extent)
    uregion = Region((M,)*3,
                     {"ADb": pygion.float32, "F_antisupport": pygion.float64})
    uregion_ups = Region((M_ups,)*3, {"F_conv_": pygion.complex128})
    prep_Fconv(uregion_ups, nonuniform_v, nonuniform_v_p,
               ac, weights, M_ups, M, N,
               reciprocal_extent, use_reciprocal_symmetry)
    right_hand(slices, slices_p, uregion, nonuniform_v, nonuniform_v_p,
               ac, weights, M, N,
               reciprocal_extent, use_reciprocal_symmetry)
    prep_Fantisupport(uregion, M)
    return uregion, uregion_ups



@task(privileges=[RO("ac"), WD("support", "estimate")])
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def phased_to_constrains(phased, ac):
    ac_smoothed = gaussian_filter(phased.ac, 0.5)
    ac.support[:] = (ac_smoothed > 1e-12).astype(np.float)
    ac.estimate[:] = phased.ac * ac.support



@task(privileges=[RO, RO, RO, WD, WD])
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def solve(uregion, uregion_ups, ac, result, summary,
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
        print('uvect_ADA =', uvect_ADA)
        uvect_FDF = autocorrelation.fourier_reg(
            uvect, ac.support, uregion.F_antisupport, M, use_reciprocal_symmetry)
        uvect = uvect_ADA + rlambda*uvect + flambda*uvect_FDF
        print('uvect in W_matvec =', uvect)
        return uvect

    def W0_matvec(uvect):
        """Define W part of the W @ x = d problem."""
        assert use_reciprocal_symmetry, "Complex AC are not supported."
        assert np.all(np.isreal(uvect))

        uvect_ADA = autocorrelation.core_problem_convolution(
            uvect, M, uregion_ups.F_conv_, M_ups, ac.support, use_reciprocal_symmetry)
        print('uvect_ADA in W0_matvec =', uvect_ADA)
        uvect = uvect_ADA 
        print('uvect in W0_matvec =', uvect)
        return uvect

    W = LinearOperator(
        dtype=np.complex128,
        shape=(M**3, M**3),
        matvec=W_matvec)

    W0 = LinearOperator(
        dtype=np.complex128,
        shape=(M**3, M**3),
        matvec=W0_matvec)

    x0 = ac.estimate.flatten()
    print('x0.dtype =', x0.dtype)
    print('x0 =', x0)
    ADb = uregion.ADb.flatten()
    print('ADb =', ADb)
    d = ADb + rlambda*x0
    d0 = ADb

    print('debug W =', W)
    print('debug d =', d)
    print('debug W0 =', W0)
    print('debug d0 =', d0)

    def callback(xk):
        callback.counter += 1
    callback.counter = 0

    ret, info = cg(W, d, x0=x0, maxiter=maxiter, callback=callback)
    print('ret.dtype =', ret.dtype)
    print('ret =', ret)
    print('info =', info)
    if info != 0:
        print(f'WARNING: CG did not converge at rlambda = {rlambda}')


    ret /= M**3
    d0 /= M**3
    soln = norm(ret)**2
    print('soln =', soln)
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    Ntot = N * N_procs # total number of pixels
    print('np.dot(ret,W0.matvec(ret)-2*d0) =', np.dot(ret,W0.matvec(ret)-2*d0))
    print('d0 =', d0)
    print('Db_squared in resid =', Db_squared)
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

    summary.rank[0] = rank
    summary.rlambda[0] = rlambda
    summary.info[0] = info
    summary.soln[0] = soln
    summary.resid[0] = resid



@task(privileges=[None, RO])
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def select_ac(generation, summary):
    print('len(summary.rank) =', len(summary.rank))
    print('summary.rank =', summary.rank)
    print('summary.rlambda =', summary.rlambda)
    print('summary.info =', summary.info)
    print('summary.soln =', summary.soln)
    print('summary.resid =', summary.resid) 
    
    # Take corner of L-curve
    valuePair = np.array([summary.resid, summary.soln]).T
    valuePair = np.array(sorted(valuePair , key=lambda k: [k[0], k[1]])) # sort the corner candidates in increasing order
    sorted_rlambdas = np.sort(summary.rlambda) # sort lambdas in increasing order
    allCoord = np.log(valuePair) # coordinates of the loglog L-curve
    nPoints = len(summary.resid)
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
    ref_rank = summary.rank[iref]

    # Log L-curve plot
    fig, axes = plt.subplots(figsize=(10.0, 24.0), nrows=3, ncols=1)
    axes[0].loglog(sorted_rlambdas, valuePair.T[1])
    axes[0].loglog(sorted_rlambdas[iref], valuePair[iref][1], "rD")
    axes[0].set_xlabel("$\lambda$")
    axes[0].set_ylabel("Solution norm $||x||_{2}$")
    axes[1].loglog(sorted_rlambdas, valuePair.T[0])
    axes[1].loglog(sorted_rlambdas[iref], valuePair[iref][0], "rD")
    axes[1].set_xlabel("$\lambda$")
    axes[1].set_ylabel("Residual norm $||Ax-b||_{2}$")
    axes[2].loglog(valuePair.T[0], valuePair.T[1]) # L-curve
    axes[2].loglog(valuePair[iref][0], valuePair[iref][1], "rD")
    axes[2].set_xlabel("Residual norm $||Ax-b||_{2}$")
    axes[2].set_ylabel("Solution norm $||x||_{2}$")
    fig.tight_layout()
    plt.savefig(parms.out_dir / f"summary_{generation}.png")
    plt.close('all')

    print(f"Keeping result from rank {ref_rank}.", flush=True)

    return iref


@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def solve_ac(generation,
             pixel_position,
             pixel_distance,
             slices,
             slices_p,
             orientations=None,
             orientations_p=None,
             phased=None):
    M = parms.M
    M_ups = parms.M_ups  # For upsampled convolution technique
    N_images_per_rank = parms.N_images_per_rank # N images per rank
    N = N_images_per_rank * utils.prod(parms.reduced_det_shape) # N images per rank x number of pixels per image = number of pixels per rank
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    N_images_tot = N_images_per_rank * N_procs
    Ntot = N * N_procs # total number of pixels
    N_pixels_per_image = N / N_images_per_rank # number of pixels per image
    reciprocal_extent = pixel_distance.reciprocal.max()
    use_reciprocal_symmetry = True
    maxiter = parms.solve_ac_maxiter

    if orientations is None:
        orientations, orientations_p = get_random_orientations()
    nonuniform, nonuniform_p = get_nonuniform_positions(
        orientations, orientations_p, pixel_position)

    ac = Region((M,)*3,
                {"support": pygion.float64, "estimate": pygion.float64})
    if phased is None:
        pygion.fill(ac, "support", 1.)
        pygion.fill(ac, "estimate", 0.)
    else:
        phased_to_constrains(phased, ac)
    weights = 1

    uregion, uregion_ups = prepare_solve(
        slices, slices_p, nonuniform, nonuniform_p,
        ac, weights, M, M_ups, N,
        reciprocal_extent, use_reciprocal_symmetry)

    Db_squared = get_Db_squared(slices, slices_p, weights, M, N)

    results = Region((N_procs * M, M, M), {"ac": pygion.float64})
    results_p = Partition.restrict(results, (N_procs,), [[M], [0], [0]], [M, M, M])

#    rlambdas = Mtot/Ntot * 1e2**(np.arange(N_procs) - N_procs/2)
    rlambdas = 1./Ntot * 10**(np.arange(N_procs) - N_procs/2)
    flambda = 0

    summary = Region((N_procs,),
                {"rank": pygion.int32, "rlambda": pygion.float32, "info": pygion.int32, "soln": pygion.float32, "resid": pygion.float32})
    summary_p = Partition.equal(summary, (N_procs,))

    for i in IndexLaunch((N_procs,)):
        solve(
            uregion, uregion_ups, ac, results_p[i], summary_p[i],
            weights, M, M_ups, N,
            generation, i, rlambdas[i], flambda, Db_squared,
            reciprocal_extent, use_reciprocal_symmetry, maxiter)

    print('uregion.ADb =', uregion.ADb)


    iref = select_ac(generation, summary)
    # At this point, I just want to chose one of the results as reference.
    # I tried to have `results` as a partition and copy into a region,
    # but I couldn't get it to work.
    return results_p[iref.get()]
