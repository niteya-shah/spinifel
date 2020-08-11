import matplotlib.pyplot as plt
import numpy as np
import pygion
import socket
from pygion import task, Region, RO, WD, Reduce, Tunable
from scipy.linalg import norm
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, cg

import pysingfel as ps

from spinifel import parms, autocorrelation, utils, image
from . import utils as lgutils


@task(privileges=[WD])
def gen_random_orientations(orientations, N_images_per_rank):
    orientations.quaternions[:] = ps.get_random_quat(N_images_per_rank)


def get_random_orientations():
    N_images_per_rank = parms.N_images_per_rank
    fields_dict = {"quaternions": pygion.float64}
    sec_shape = (4,)
    orientations, orientations_p = lgutils.create_distributed_region(
        N_images_per_rank, fields_dict, sec_shape)
    for i, orientations_subr in enumerate(orientations_p):
        gen_random_orientations(orientations_subr, N_images_per_rank, point=i)
    return orientations, orientations_p


@task(privileges=[RO, WD])
def gen_nonuniform_positions_v(nonuniform, nonuniform_v, reciprocal_extent):
    nonuniform_v.H[:] = (nonuniform.H.flatten()
        / reciprocal_extent * np.pi / parms.oversampling)
    nonuniform_v.K[:] = (nonuniform.K.flatten()
        / reciprocal_extent * np.pi / parms.oversampling)
    nonuniform_v.L[:] = (nonuniform.L.flatten()
        / reciprocal_extent * np.pi / parms.oversampling)


def get_nonuniform_positions_v(nonuniform, nonuniform_p, reciprocal_extent):
    """Flatten and calibrate nonuniform positions."""
    N_vals_per_rank = (
        parms.N_images_per_rank * utils.prod(parms.reduced_det_shape))
    fields_dict = {"H": pygion.float64, "K": pygion.float64,
                   "L": pygion.float64}
    sec_shape = ()
    nonuniform_v, nonuniform_v_p = lgutils.create_distributed_region(
        N_vals_per_rank, fields_dict, sec_shape)
    for i, (nonuniform_subr, nonuniform_v_subr) in enumerate(zip(
            nonuniform_p, nonuniform_v_p)):
        gen_nonuniform_positions_v(nonuniform_subr, nonuniform_v_subr,
                                   reciprocal_extent, point=i)
        # Ideally, the location (point) should be deduced from the
        # location of the slices.
    return nonuniform_v, nonuniform_v_p


@task(privileges=[RO, WD, RO])
def gen_nonuniform_positions(orientations, nonuniform, pixel_position):
    H, K, L = autocorrelation.gen_nonuniform_positions(
        orientations.quaternions, pixel_position.reciprocal)
    nonuniform.H[:] = H
    nonuniform.K[:] = K
    nonuniform.L[:] = L


def get_nonuniform_positions(orientations, orientations_p, pixel_position):
    N_images_per_rank = parms.N_images_per_rank
    fields_dict = {"H": pygion.float64, "K": pygion.float64,
                   "L": pygion.float64}
    sec_shape = parms.reduced_det_shape
    nonuniform, nonuniform_p = lgutils.create_distributed_region(
        N_images_per_rank, fields_dict, sec_shape)
    for i, (orientations_subr, nonuniform_subr) in enumerate(zip(
            orientations_p, nonuniform_p)):
        gen_nonuniform_positions(
            orientations_subr, nonuniform_subr, pixel_position, point=i)
        # Ideally, the location (point) should be deduced from the
        # location of the slices.
    return nonuniform, nonuniform_p


@task(privileges=[RO, Reduce('+', 'ADb'), RO, RO])
def right_hand_ADb_task(slices, uregion, nonuniform_v, ac, weights, M,
                        reciprocal_extent, use_reciprocal_symmetry):
    data = slices.data.flatten()
    nuvect_Db = data * weights
    uregion.ADb[:] += autocorrelation.adjoint(
        nuvect_Db,
        nonuniform_v.H,
        nonuniform_v.K,
        nonuniform_v.L,
        ac.support, M,
        reciprocal_extent, use_reciprocal_symmetry
    )
    print(f"{socket.gethostname()} computed ADb.", flush=True)


def right_hand(slices, slices_p, uregion, nonuniform_v, nonuniform_v_p,
               ac, weights, M,
               reciprocal_extent, use_reciprocal_symmetry):
    pygion.fill(uregion, "ADb", 0.)
    for i, (slices_subr, nonuniform_v_subr) in enumerate(zip(
            slices_p, nonuniform_v_p)):
        right_hand_ADb_task(slices_subr, uregion, nonuniform_v_subr,
                            ac, weights, M,
                            reciprocal_extent, use_reciprocal_symmetry,
                            point=i)
        # Ideally, the location (point) should be deduced from the
        # location of the slices.


@task(privileges=[Reduce('+', 'F_conv_'), RO, RO])
def prep_Fconv_task(uregion_ups, nonuniform_v, ac, weights, M_ups, Mtot, N,
                    reciprocal_extent, use_reciprocal_symmetry):
    conv_ups = autocorrelation.adjoint(
        np.ones(N),
        nonuniform_v.H,
        nonuniform_v.K,
        nonuniform_v.L,
        1, M_ups,
        reciprocal_extent, use_reciprocal_symmetry
    )
    uregion_ups.F_conv_[:] += np.fft.fftn(np.fft.ifftshift(conv_ups)) / Mtot
    print(f"{socket.gethostname()} computed Fconv.", flush=True)


def prep_Fconv(uregion_ups, nonuniform_v, nonuniform_v_p,
               ac, weights, M_ups, Mtot, N,
               reciprocal_extent, use_reciprocal_symmetry):
    pygion.fill(uregion_ups, "F_conv_", 0.)
    for i, nonuniform_v_subr in enumerate(nonuniform_v_p):
        prep_Fconv_task(uregion_ups, nonuniform_v_subr,
                        ac, weights, M_ups, Mtot, N,
                        reciprocal_extent, use_reciprocal_symmetry,
                        point=i)
        # Ideally, the location (point) should be deduced from the
        # location of the slices.


@task(privileges=[WD("F_antisupport")])
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


def prepare_solve(slices, slices_p, nonuniform, nonuniform_p,
                  ac, weights, M, Mtot, M_ups, N,
                  reciprocal_extent, use_reciprocal_symmetry):
    nonuniform_v, nonuniform_v_p = get_nonuniform_positions_v(
        nonuniform, nonuniform_p, reciprocal_extent)
    uregion = Region((M,)*3,
                     {"ADb": pygion.float64, "F_antisupport": pygion.float32})
    uregion_ups = Region((M_ups,)*3, {"F_conv_": pygion.complex128})
    prep_Fconv(uregion_ups, nonuniform_v, nonuniform_v_p,
               ac, weights, M_ups, Mtot, N,
               reciprocal_extent, use_reciprocal_symmetry)
    right_hand(slices, slices_p, uregion, nonuniform_v, nonuniform_v_p,
               ac, weights, M,
               reciprocal_extent, use_reciprocal_symmetry)
    prep_Fantisupport(uregion, M)
    return uregion, uregion_ups


@task(privileges=[RO("ac"), WD("support", "estimate")])
def phased_to_constrains(phased, ac):
    ac_smoothed = gaussian_filter(phased.ac, 0.5)
    ac.support[:] = (ac_smoothed > 1e-12).astype(np.float)
    ac.estimate[:] = phased.ac * ac.support


@task(privileges=[RO, RO, RO, WD])
def solve(uregion, uregion_ups, ac, result,
          weights, M, M_ups, Mtot, N,
          generation, rank, alambda, rlambda, flambda,
          reciprocal_extent, use_reciprocal_symmetry):
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
        dtype=np.complex128,
        shape=(Mtot, Mtot),
        matvec=W_matvec)

    x0 = ac.estimate.flatten()
    ADb = uregion.ADb.flatten()
    d = alambda*ADb + rlambda*x0

    maxiter = 100

    def callback(xk):
        callback.counter += 1
    callback.counter = 0

    ret, info = cg(W, d, x0=x0, maxiter=maxiter, callback=callback)
    ac_res = ret.reshape((M,)*3)
    if use_reciprocal_symmetry:
        assert np.all(np.isreal(ac_res))
    result.ac[:] = ac_res.real
    it_number = callback.counter

    print(f"{socket.gethostname()} - gen {generation} - rank {rank} recovered AC in {it_number} iterations.", flush=True)
    image.show_volume(result.ac[:], parms.Mquat,
                      f"autocorrelation_{generation}_{rank}.png")

    v1 = norm(ret)
    v2 = norm(W*ret-d)
    return rank, rlambda, v1, v2


@task
def select_ac(generation, *summary):
    summary = [el.get() for el in summary]
    ranks, lambdas, v1s, v2s = [np.array(el) for el in zip(*summary)]

    if generation == 0:
        # Expect non-convergence => weird results.
        # Heuristic: retain rank with highest lambda and high v1.
        idx = v1s >= np.mean(v1s)
        imax = np.argmax(lambdas[idx])
        iref = np.arange(len(ranks), dtype=np.int)[idx][imax]
    else:
        # Take corner of L-curve: min (v1+v2)
        iref = np.argmin(v1s+v2s)
    ref_rank = ranks[iref]

    fig, axes = plt.subplots(figsize=(6.0, 6.0), nrows=2, ncols=1)
    axes[0].semilogx(lambdas, v1s)
    axes[0].semilogx(lambdas[iref], v1s[iref], "rD")
    axes[0].set_xlabel("$\\lambda_r$")
    axes[0].set_ylabel("$\\|x\\|$")
    axes[1].semilogx(lambdas, v2s)
    axes[1].semilogx(lambdas[iref], v2s[iref], "rD")
    axes[1].set_xlabel("$\\lambda_r$")
    axes[1].set_ylabel("$\\|W x - d\\|$")
    plt.savefig(parms.out_dir / f"summary_{generation}.png")
    plt.close('all')

    print(f"Keeping result from rank {ref_rank}.", flush=True)

    return iref


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
    Mtot = M**3
    N_images_per_rank = parms.N_images_per_rank
    N = N_images_per_rank * utils.prod(parms.reduced_det_shape)
    reciprocal_extent = pixel_distance.reciprocal.max()
    use_reciprocal_symmetry = True

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

    N_ranks = 5
    results = [Region((M,)*3, {"ac": pygion.float64}) for i in range(N_ranks)]

    alambda = 1
    rlambdas = 1e-7 * 100**np.arange(N_ranks)
    flambda = 0
    summary = []

    for i in range(N_ranks):
        summary.append(solve(
            uregion, uregion_ups, ac, results[i],
            weights, M, M_ups, Mtot, N,
            generation, i, alambda, rlambdas[i], flambda,
            reciprocal_extent, use_reciprocal_symmetry, point=i%N_procs))
        # Ideally, the mapper should distribute these tasks. However,
        # here, without the point, everything runs on the same node.

    iref = select_ac(generation, *summary)
    # At this point, I just want to chose one of the results as reference.
    # I tried to have `results` as a partition and copy into a region,
    # but I couldn't get it to work.
    return results[iref.get()]
