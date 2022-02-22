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



@task(privileges=[WD])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def gen_random_orientations(orientations, N_images_per_rank):
    orientations.quaternions[:] = skp.get_random_quat(N_images_per_rank)



@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def get_random_orientations():
    N_images_per_rank = settings.N_images_per_rank
    fields_dict = {"quaternions": pygion.float32}
    sec_shape = (4,)
    orientations, orientations_p = lgutils.create_distributed_region(
        N_images_per_rank, fields_dict, sec_shape)
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in IndexLaunch([N_procs]):
        gen_random_orientations(orientations_p[i], N_images_per_rank)
    return orientations, orientations_p



@task(privileges=[RO, WD])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def gen_nonuniform_positions_v(nonuniform, nonuniform_v, reciprocal_extent):
    nonuniform_v.H[:] = (nonuniform.H.flatten()
        / reciprocal_extent * np.pi / settings.oversampling)
    nonuniform_v.K[:] = (nonuniform.K.flatten()
        / reciprocal_extent * np.pi / settings.oversampling)
    nonuniform_v.L[:] = (nonuniform.L.flatten()
        / reciprocal_extent * np.pi / settings.oversampling)



@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def get_nonuniform_positions_v(nonuniform, nonuniform_p, reciprocal_extent):
    """Flatten and calibrate nonuniform positions."""
    N_vals_per_rank = (
        settings.N_images_per_rank * utils.prod(settings.reduced_det_shape))
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
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def gen_nonuniform_positions(orientations, nonuniform, pixel_position):
    H, K, L = autocorrelation.gen_nonuniform_positions(
        orientations.quaternions, pixel_position.reciprocal)
    nonuniform.H[:] = H
    nonuniform.K[:] = K
    nonuniform.L[:] = L



@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def get_nonuniform_positions(orientations, orientations_p, pixel_position):
    N_images_per_rank = settings.N_images_per_rank
    fields_dict = {"H": pygion.float32, "K": pygion.float32,
                   "L": pygion.float32}
    sec_shape = settings.reduced_det_shape
    nonuniform, nonuniform_p = lgutils.create_distributed_region(
        N_images_per_rank, fields_dict, sec_shape)
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in IndexLaunch([N_procs]):
        gen_nonuniform_positions(
            orientations_p[i], nonuniform_p[i], pixel_position)
    return nonuniform, nonuniform_p



@task(privileges=[RO, Reduce('+', 'ADb'), RO, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def right_hand_ADb_task(slices, uregion, nonuniform_v, ac, weights, M,
                        reciprocal_extent, use_reciprocal_symmetry):
    if settings.verbosity > 0:
        print(f"{socket.gethostname()} started ADb.", flush=True)
    data = slices.data.flatten()
    nuvect_Db = data * weights
    uregion.ADb[:] += autocorrelation.adjoint(
        nuvect_Db,
        nonuniform_v.H,
        nonuniform_v.K,
        nonuniform_v.L,
        M,
        support=ac.support,
        use_recip_sym=use_reciprocal_symmetry)

    if settings.verbosity > 0:
        print(f"{socket.gethostname()} computed ADb.", flush=True)



@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def right_hand(slices, slices_p, uregion, nonuniform_v, nonuniform_v_p,
               ac, weights, M,
               reciprocal_extent, use_reciprocal_symmetry):
    pygion.fill(uregion, "ADb", 0.)
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in IndexLaunch([N_procs]):
        right_hand_ADb_task(slices_p[i], uregion, nonuniform_v_p[i],
                            ac, weights, M,
                            reciprocal_extent, use_reciprocal_symmetry)



@task(privileges=[Reduce('+', 'F_conv_'), RO, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def prep_Fconv_task(uregion_ups, nonuniform_v, ac, weights, M_ups, Mtot, N,
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
    uregion_ups.F_conv_[:] += np.fft.fftn(np.fft.ifftshift(conv_ups)) / Mtot
    if settings.verbosity > 0:
        print(f"{socket.gethostname()} computed Fconv.", flush=True)



@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def prep_Fconv(uregion_ups, nonuniform_v, nonuniform_v_p,
               ac, weights, M_ups, Mtot, N,
               reciprocal_extent, use_reciprocal_symmetry):
    pygion.fill(uregion_ups, "F_conv_", 0.)
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    for i in IndexLaunch([N_procs]):
        prep_Fconv_task(uregion_ups, nonuniform_v_p[i],
                        ac, weights, M_ups, Mtot, N,
                        reciprocal_extent, use_reciprocal_symmetry)



@task(privileges=[WD("F_antisupport")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def prep_Fantisupport(uregion, M):
    lu = np.linspace(-np.pi, np.pi, M)
    Hu_, Ku_, Lu_ = np.meshgrid(lu, lu, lu, indexing='ij')
    Qu_ = np.sqrt(Hu_**2 + Ku_**2 + Lu_**2)
    uregion.F_antisupport[:] = Qu_ > np.pi / settings.oversampling

    Fantisup = uregion.F_antisupport
    assert np.all(Fantisup[:] == Fantisup[::-1, :, :])
    assert np.all(Fantisup[:] == Fantisup[:, ::-1, :])
    assert np.all(Fantisup[:] == Fantisup[:, :, ::-1])
    assert np.all(Fantisup[:] == Fantisup[::-1, ::-1, ::-1])



@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def prepare_solve(slices, slices_p, nonuniform, nonuniform_p,
                  ac, weights, M, Mtot, M_ups, N,
                  reciprocal_extent, use_reciprocal_symmetry):
    nonuniform_v, nonuniform_v_p = get_nonuniform_positions_v(
        nonuniform, nonuniform_p, reciprocal_extent)
    uregion = Region((M,)*3,
                     {"ADb": pygion.float32, "F_antisupport": pygion.float32})
    uregion_ups = Region((M_ups,)*3, {"F_conv_": pygion.complex64})
    prep_Fconv(uregion_ups, nonuniform_v, nonuniform_v_p,
               ac, weights, M_ups, Mtot, N,
               reciprocal_extent, use_reciprocal_symmetry)
    right_hand(slices, slices_p, uregion, nonuniform_v, nonuniform_v_p,
               ac, weights, M,
               reciprocal_extent, use_reciprocal_symmetry)
    prep_Fantisupport(uregion, M)
    return uregion, uregion_ups



@task(privileges=[RO("ac"), WD("support", "estimate")])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def phased_to_constrains(phased, ac):
    ac_smoothed = gaussian_filter(phased.ac, 0.5)
    ac.support[:] = (ac_smoothed > 1e-12).astype(np.float)
    ac.estimate[:] = phased.ac * ac.support



@task(privileges=[RO, RO, RO, WD, WD])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def solve(uregion, uregion_ups, ac, result, summary,
          weights, M, M_ups, Mtot, N,
          generation, rank, alambda, rlambda, flambda,
          reciprocal_extent, use_reciprocal_symmetry, maxiter):
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

    W = LinearOperator(
        dtype=np.complex64,
        shape=(Mtot, Mtot),
        matvec=W_matvec)

    x0 = ac.estimate.flatten()
    ADb = uregion.ADb.flatten()
    d = alambda*ADb + rlambda*x0

    def callback(xk):
        callback.counter += 1
    callback.counter = 0

    ret, info = cg(W, d, x0=x0, maxiter=maxiter, callback=callback)
    ac_res = ret.reshape((M,)*3)
    if use_reciprocal_symmetry:
        assert np.all(np.isreal(ac_res))
    result.ac[:] = ac_res.real
    it_number = callback.counter

    if settings.verbosity > 0:
        print(f"{socket.gethostname()} - ", end='')
    print(f"Rank {rank} recovered AC in {it_number} iterations.", flush=True)
    image.show_volume(result.ac[:], settings.Mquat,
                      f"autocorrelation_{generation}_{rank}.png")

    v1 = norm(ret)
    v2 = norm(W*ret-d)

    summary.rank[0] = rank
    summary.rlambda[0] = rlambda
    summary.v1[0] = v1
    summary.v2[0] = v2



@task(privileges=[None, RO])
@lgutils.gpu_task_wrapper
@nvtx.annotate("legion/autocorrelation.py", is_prefix=True)
def select_ac(generation, summary):
    if generation == 0:
        # Expect non-convergence => weird results.
        # Heuristic: retain rank with highest lambda and high v1.
        idx = summary.v1 >= np.mean(summary.v1)
        imax = np.argmax(summary.rlambda[idx])
        iref = np.arange(len(summary.rank), dtype=np.int)[idx][imax]
    else:
        # Take corner of L-curve: min (v1+v2)
        iref = np.argmin(summary.v1+summary.v2)
    ref_rank = summary.rank[iref]

    fig, axes = plt.subplots(figsize=(6.0, 8.0), nrows=3, ncols=1)
    axes[0].loglog(summary.rlambda, summary.v1)
    axes[0].loglog(summary.rlambda[iref], summary.v1[iref], "rD")
    axes[0].set_xlabel("$\lambda_{r}$")
    axes[0].set_ylabel("$||x_{\lambda_{r}}||_{2}$")
    axes[1].loglog(summary.rlambda, summary.v2)
    axes[1].loglog(summary.rlambda[iref], summary.v2[iref], "rD")
    axes[1].set_xlabel("$\lambda_{r}$")
    axes[1].set_ylabel("$||W \lambda_{r}-d||_{2}$")
    axes[2].loglog(summary.v2, summary.v1) # L-curve
    axes[2].loglog(summary.v2[iref], summary.v1[iref], "rD")
    axes[2].set_xlabel("Residual norm $||W \lambda_{r}-d||_{2}$")
    axes[2].set_ylabel("Solution norm $||x_{\lambda_{r}}||_{2}$")
    fig.tight_layout()
    plt.savefig(settings.out_dir / f"summary_{generation}.png")
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
    M = settings.M
    M_ups = settings.M_ups  # For upsampled convolution technique
    Mtot = M**3
    N_images_per_rank = settings.N_images_per_rank
    N = N_images_per_rank * utils.prod(settings.reduced_det_shape)
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    N_images_tot = N_images_per_rank * N_procs
    Ntot = N * N_procs
    reciprocal_extent = pixel_distance.reciprocal.max()
    use_reciprocal_symmetry = True
    maxiter = settings.solve_ac_maxiter

    if orientations is None:
        orientations, orientations_p = get_random_orientations()
    nonuniform, nonuniform_p = get_nonuniform_positions(
        orientations, orientations_p, pixel_position)

    ac = Region((M,)*3,
                {"support": pygion.float32, "estimate": pygion.float32})
    if phased is None:
        pygion.fill(ac, "support", 1.)
        pygion.fill(ac, "estimate", 0.)
    else:
        phased_to_constrains(phased, ac)
    weights = 1

    uregion, uregion_ups = prepare_solve(
        slices, slices_p, nonuniform, nonuniform_p,
        ac, weights, M, Mtot, M_ups, N,
        reciprocal_extent, use_reciprocal_symmetry)

    #N_ranks = 5
    N_procs = Tunable.select(Tunable.GLOBAL_PYS).get()
    results = Region((N_procs * M, M, M), {"ac": pygion.float32})
    results_p = Partition.restrict(results, (N_procs,), [[M], [0], [0]], [M, M, M])

    alambda = 1
#    rlambdas = Mtot/Ntot * 1e2**(np.arange(N_procs) - N_procs/2)
    rlambdas = Mtot/Ntot * 2**(np.arange(N_procs) - N_procs/2)
    flambda = 0

    summary = Region((N_procs,),
                {"rank": pygion.int32, "rlambda": pygion.float32, "v1": pygion.float32, "v2": pygion.float32})
    summary_p = Partition.equal(summary, (N_procs,))


    for i in IndexLaunch((N_procs,)):
        solve(
            uregion, uregion_ups, ac, results_p[i], summary_p[i],
            weights, M, M_ups, Mtot, N,
            generation, i, alambda, rlambdas[i], flambda,
            reciprocal_extent, use_reciprocal_symmetry, maxiter)

    iref = select_ac(generation, summary)
    # At this point, I just want to chose one of the results as reference.
    # I tried to have `results` as a partition and copy into a region,
    # but I couldn't get it to work.
    return results_p[iref.get()]

