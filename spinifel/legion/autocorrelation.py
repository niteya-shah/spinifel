import numpy as np
import pygion
from pygion import task, Region, RO, WD, Reduce, Tunable
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
    for orientations_subr in orientations_p:
        gen_random_orientations(orientations_subr, N_images_per_rank)
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
    for nonuniform_subr, nonuniform_v_subr in zip(
            nonuniform_p, nonuniform_v_p):
        gen_nonuniform_positions_v(nonuniform_subr, nonuniform_v_subr,
                                   reciprocal_extent)
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
    for orientations_subr, nonuniform_subr in zip(
            orientations_p, nonuniform_p):
        gen_nonuniform_positions(
            orientations_subr, nonuniform_subr, pixel_position)
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


def right_hand(slices, slices_p, uregion, nonuniform_v, nonuniform_v_p,
               ac, weights, M,
               reciprocal_extent, use_reciprocal_symmetry):
    pygion.fill(uregion, "ADb", 0.)
    for slices_subr, nonuniform_v_subr in zip(slices_p, nonuniform_v_p):
        right_hand_ADb_task(slices_subr, uregion, nonuniform_v_subr,
                            ac, weights, M,
                            reciprocal_extent, use_reciprocal_symmetry)


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


def prep_Fconv(uregion_ups, nonuniform_v, nonuniform_v_p,
               ac, weights, M_ups, Mtot, N,
               reciprocal_extent, use_reciprocal_symmetry):
    pygion.fill(uregion_ups, "F_conv_", 0.)
    for nonuniform_v_subr in nonuniform_v_p:
        prep_Fconv_task(uregion_ups, nonuniform_v_subr,
                        ac, weights, M_ups, Mtot, N,
                        reciprocal_extent, use_reciprocal_symmetry)


@task(privileges=[RO, RO, RO])
def solve(uregion, uregion_ups, ac,
          weights, M, M_ups, Mtot, N,
          generation,
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
        uvect = uvect_ADA
        return uvect

    W = LinearOperator(
        dtype=np.complex128,
        shape=(Mtot, Mtot),
        matvec=W_matvec)

    x0 = ac.estimate.flatten()
    d = uregion.ADb.flatten()

    maxiter = 100

    def callback(xk):
        callback.counter += 1
    callback.counter = 0

    ret, info = cg(W, d, x0=x0, maxiter=maxiter, callback=callback)
    ac_res = ret.reshape((M,)*3)
    if use_reciprocal_symmetry:
        assert np.all(np.isreal(ac_res))
    ac_res = ac_res.real
    it_number = callback.counter

    print(f"Recovered AC in {it_number} iterations.", flush=True)
    image.show_volume(ac_res, parms.Mquat, f"autocorrelation_{generation}.png")


def solve_ac(generation,
             pixel_position,
             pixel_distance,
             slices,
             slices_p,
             orientations=None,
             orientations_p=None,
             ac_estimate=None):
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

    if ac_estimate is None:
        ac = Region((M,)*3,
                    {"support": pygion.float32, "estimate": pygion.float32})
        pygion.fill(ac, "support", 1.)
        pygion.fill(ac, "estimate", 0.)
    else:
        raise NotImplemented()
    weights = 1

    # BEGIN Setup Linear Operator

    nonuniform_v, nonuniform_v_p = get_nonuniform_positions_v(
        nonuniform, nonuniform_p, reciprocal_extent)
    uregion = Region((M,)*3, {"ADb": pygion.float64})
    uregion_ups = Region((M_ups,)*3, {"F_conv_": pygion.complex128})

    prep_Fconv(uregion_ups, nonuniform_v, nonuniform_v_p,
               ac, weights, M_ups, Mtot, N,
               reciprocal_extent, use_reciprocal_symmetry)

    right_hand(slices, slices_p, uregion, nonuniform_v, nonuniform_v_p,
               ac, weights, M,
               reciprocal_extent, use_reciprocal_symmetry)

    print("WARNING: Legion implementation of AC solver is incomplete.")

    # END Setup Linear Operator

    solve(uregion, uregion_ups, ac,
          weights, M, M_ups, Mtot, N,
          generation,
          reciprocal_extent, use_reciprocal_symmetry)
