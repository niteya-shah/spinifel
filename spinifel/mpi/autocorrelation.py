from mpi4py import MPI

import finufftpy as nfft
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, cg

import pysingfel as ps

from spinifel import parms, utils, image, autocorrelation


def reduce_bcast(comm, vect):
    vect = np.ascontiguousarray(vect)
    reduced_vect = np.zeros_like(vect)
    comm.Reduce(vect, reduced_vect, op=MPI.SUM, root=0)
    vect = reduced_vect
    comm.Bcast(vect, root=0)
    return vect


def core_problem(comm, uvect, H_, K_, L_, ac_support, weights, M, N,
                 reciprocal_extent, use_reciprocal_symmetry):
    comm.Bcast(uvect, root=0)
    uvect_ADA = autocorrelation.core_problem(
        uvect, H_, K_, L_, ac_support, weights, M, N,
        reciprocal_extent, use_reciprocal_symmetry)
    uvect_ADA = reduce_bcast(comm, uvect_ADA)
    return uvect_ADA


def fourier_reg(uvect, support, F_antisupport, M, use_recip_sym):
    ugrid = uvect.reshape((M,)*3) * support
    if use_recip_sym:
        assert np.all(np.isreal(ugrid))
    F_ugrid = np.fft.fftn(np.fft.ifftshift(ugrid))
    F_reg = F_ugrid * np.fft.ifftshift(F_antisupport)
    reg = np.fft.fftshift(np.fft.ifftn(F_reg))
    uvect = (reg * support).flatten()
    if use_recip_sym:
        uvect = uvect.real
    return uvect


def setup_linops(comm, H, K, L, data,
                 ac_support, weights, x0,
                 M, Mtot, N, reciprocal_extent,
                 alambda, rlambda, flambda,
                 use_reciprocal_symmetry):
    """Define W and d parts of the W @ x = d problem.

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
    H_ = H.flatten() / reciprocal_extent * np.pi / parms.oversampling
    K_ = K.flatten() / reciprocal_extent * np.pi / parms.oversampling
    L_ = L.flatten() / reciprocal_extent * np.pi / parms.oversampling

    lu = np.linspace(-np.pi, np.pi, M)
    Hu_, Ku_, Lu_ = np.meshgrid(lu, lu, lu, indexing='ij')
    Qu_ = np.sqrt(Hu_**2 + Ku_**2 + Lu_**2)
    F_antisupport = Qu_ > np.pi / parms.oversampling
    assert np.all(F_antisupport == F_antisupport[::-1, :, :])
    assert np.all(F_antisupport == F_antisupport[:, ::-1, :])
    assert np.all(F_antisupport == F_antisupport[:, :, ::-1])
    assert np.all(F_antisupport == F_antisupport[::-1, ::-1, ::-1])

    def W_matvec(uvect):
        """Define W part of the W @ x = d problem."""
        uvect_ADA = core_problem(  # A_adj*Da*A
            comm, uvect, H_, K_, L_, ac_support, weights, M, N,
            reciprocal_extent, use_reciprocal_symmetry)
        uvect_FDF = fourier_reg(  # A_adj*Da*A
            uvect, ac_support, F_antisupport, M, use_reciprocal_symmetry)
        uvect = alambda*uvect_ADA + rlambda*uvect + flambda*uvect_FDF
        return uvect

    W = LinearOperator(
        dtype=np.complex128,
        shape=(Mtot, Mtot),
        matvec=W_matvec)

    nuvect_Db = data * weights
    uvect_ADb = autocorrelation.adjoint(
        nuvect_Db, H_, K_, L_, ac_support, M,
        reciprocal_extent, use_reciprocal_symmetry)
    uvect_ADb = reduce_bcast(comm, uvect_ADb)
    d = alambda*uvect_ADb + rlambda*x0

    return W, d


def solve_ac(generation,
             pixel_position_reciprocal,
             pixel_distance_reciprocal,
             slices_,
             orientations=None,
             ac_estimate=None):
    comm = MPI.COMM_WORLD

    M = parms.M
    Mtot = M**3
    N_images = slices_.shape[0]
    N = utils.prod(slices_.shape)
    reciprocal_extent = pixel_distance_reciprocal.max()
    use_reciprocal_symmetry = True

    if orientations is None:
        orientations = ps.get_random_quat(N_images)
    H, K, L = autocorrelation.gen_nonuniform_positions(
        orientations, pixel_position_reciprocal)

    data = slices_.flatten()

    if ac_estimate is None:
        ac_support = np.ones((M,)*3)
        ac_estimate = np.zeros((M,)*3)
    else:
        ac_smoothed = gaussian_filter(ac_estimate, 0.5)
        ac_support = (ac_smoothed > 1e-12).astype(np.float)
        ac_estimate *= ac_support
    weights = np.ones(N)

    alambda = 1
    rlambda = 1e-9
    flambda = 1e3
    maxiter = 100

    if comm.rank == 0:
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

    x0 = ac_estimate.flatten()
    W, d = setup_linops(comm, H, K, L, data,
                        ac_support, weights, x0,
                        M, Mtot, N, reciprocal_extent,
                        alambda, rlambda, flambda,
                        use_reciprocal_symmetry)
    ret, info = cg(W, d, x0=x0, maxiter=maxiter, callback=callback)
    ac = ret.reshape((M,)*3)
    assert np.all(np.isreal(ac))  # if use_reciprocal_symmetry
    ac = ac.real
    it_number = callback.counter

    print(f"Rank {comm.rank} got AC in {it_number} iterations.", flush=True)
    if comm.rank == 0:
        image.show_volume(ac, parms.Mquat, f"autocorrelation_{generation}.png")

    return ac
