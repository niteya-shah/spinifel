import finufftpy as nfft
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.sparse.linalg import LinearOperator, cg

import pysingfel as ps

from spinifel import parms, utils


def forward(uvect, H, K, L, support, M, N,
            recip_extent, use_recip_sym):
    """Apply the forward, NUFFT2- problem."""
    if use_recip_sym:
        assert np.all(np.isreal(uvect))
    ugrid = uvect.reshape((M,)*3) * support
    nuvect = np.zeros(N, dtype=np.complex)
    assert not nfft.nufft3d2(
        H/recip_extent*np.pi,
        K/recip_extent*np.pi,
        L/recip_extent*np.pi,
        nuvect, -1, 1e-12, ugrid)
    return nuvect / M**3


def adjoint(nuvect, H, K, L, support, M,
            recip_extent, use_recip_sym):
    """Apply the adjoint, NUFFT1+ problem."""
    ugrid = np.zeros((M,)*3, dtype=np.complex, order='F')
    assert not nfft.nufft3d1(
        H/recip_extent*np.pi,
        K/recip_extent*np.pi,
        L/recip_extent*np.pi,
        nuvect, +1, 1e-12, M, M, M, ugrid)
    uvect = (ugrid * support).flatten()
    if use_recip_sym:
        uvect = uvect.real
    return uvect


def solve_ac(pixel_position_reciprocal,
             pixel_distance_reciprocal,
             slices_):
    Mquat = 10
    M = 4 * Mquat + 1
    Mtot = M**3
    N_images = slices_.shape[0]
    N = utils.prod(slices_.shape)
    real_extent = 2
    reciprocal_extent = pixel_distance_reciprocal.max()
    use_reciprocal_symmetry = True

    orientations = ps.get_random_quat(N_images)
    rotmat = np.array([ps.quaternion2rot3d(quat) for quat in orientations])
    H, K, L = np.einsum("ijk,klmn->jilmn", rotmat, pixel_position_reciprocal)
    # shape -> [N_images] x det_shape

    data = slices_.flatten()
    Hf = H.flatten()
    Kf = K.flatten()
    Lf = L.flatten()

    ac_support = np.ones((M,)*3)
    ac_estimate = np.zeros((M,)*3)
    weights = np.ones(N)

    alambda = 1
    rlambda = 1e-9
    maxiter = 100

    idx = np.abs(L) < reciprocal_extent * .01
    plt.scatter(H[idx], K[idx], c=slices_[idx], s=1, norm=LogNorm())
    plt.axis('equal')
    plt.colorbar()
    plt.savefig(parms.out_dir / "star_0.png")
    plt.cla()
    plt.clf()

    A = LinearOperator(
        dtype=np.complex128,
        shape=(N, Mtot),
        matvec=lambda x: forward(
            x, H, K, L, ac_support, M, N,
            reciprocal_extent, use_reciprocal_symmetry))

    A_adj = LinearOperator(
        dtype=np.complex128,
        shape=(Mtot, N),
        matvec=lambda x: adjoint(
            x, H, K, L, ac_support, M,
            reciprocal_extent, use_reciprocal_symmetry))

    I = LinearOperator(
        dtype=np.complex128,
        shape=(Mtot, Mtot),
        matvec=lambda x: x)

    D = LinearOperator(
        dtype=np.complex128,
        shape=(N, N),
        matvec=lambda x: weights*x,
        rmatvec=lambda x: weights*x)

    A._adjoint = lambda: A_adj
    A_adj._adjoint = lambda: A
    I._adjoint = lambda: I

    def callback(xk):
        callback.counter += 1
        print('+', end='')
    callback.counter = 0

    x0 = ac_estimate.flatten()
    b = data
    al = alambda
    rl = rlambda
    W = al * A.H * D * A + rl * I
    d = al * A.H * D * b + rl * x0
    ret, info = cg(W, d, x0=x0, maxiter=maxiter, callback=callback)
    print('')
    return ret.reshape((M,)*3), callback.counter
