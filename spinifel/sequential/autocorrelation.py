import finufftpy as nfft
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, cg

import pysingfel as ps

from spinifel import parms, utils, image


def forward(uvect, H_, K_, L_, support, M, N,
            recip_extent, use_recip_sym):
    """Apply the forward, NUFFT2- problem."""
    if use_recip_sym:
        assert np.all(np.isreal(uvect))
    ugrid = uvect.reshape((M,)*3) * support
    nuvect = np.zeros(N, dtype=np.complex)
    assert not nfft.nufft3d2(
        H_,
        K_,
        L_,
        nuvect, -1, 1e-12, ugrid)
    return nuvect / M**3


def adjoint(nuvect, H_, K_, L_, support, M,
            recip_extent, use_recip_sym):
    """Apply the adjoint, NUFFT1+ problem."""
    ugrid = np.zeros((M,)*3, dtype=np.complex, order='F')
    assert not nfft.nufft3d1(
        H_,
        K_,
        L_,
        nuvect, +1, 1e-12, M, M, M, ugrid)
    uvect = (ugrid * support).flatten()
    if use_recip_sym:
        uvect = uvect.real
    return uvect


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


def solve_ac(generation,
             pixel_position_reciprocal,
             pixel_distance_reciprocal,
             slices_,
             orientations=None,
             ac_estimate=None):
    M = parms.M
    Mtot = M**3
    N_images = slices_.shape[0]
    N = utils.prod(slices_.shape)
    reciprocal_extent = pixel_distance_reciprocal.max()
    use_reciprocal_symmetry = True

    if orientations is None:
        orientations = ps.get_random_quat(N_images)
    rotmat = np.array([ps.quaternion2rot3d(quat) for quat in orientations])
    H, K, L = np.einsum("ijk,klmn->jilmn", rotmat, pixel_position_reciprocal)
    # shape -> [N_images] x det_shape

    data = slices_.flatten()
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

    idx = np.abs(L) < reciprocal_extent * .01
    plt.scatter(H[idx], K[idx], c=slices_[idx], s=1, norm=LogNorm())
    plt.axis('equal')
    plt.colorbar()
    plt.savefig(parms.out_dir / f"star_{generation}.png")
    plt.cla()
    plt.clf()

    A = LinearOperator(
        dtype=np.complex128,
        shape=(N, Mtot),
        matvec=lambda x: forward(
            x, H_, K_, L_, ac_support, M, N,
            reciprocal_extent, use_reciprocal_symmetry))

    A_adj = LinearOperator(
        dtype=np.complex128,
        shape=(Mtot, N),
        matvec=lambda x: adjoint(
            x, H_, K_, L_, ac_support, M,
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

    F_reg = LinearOperator(
        dtype=np.complex128,
        shape=(Mtot, Mtot),
        matvec=lambda x: fourier_reg(
            x, ac_support, F_antisupport, M, use_reciprocal_symmetry))

    A._adjoint = lambda: A_adj
    A_adj._adjoint = lambda: A
    I._adjoint = lambda: I

    def callback(xk):
        callback.counter += 1
    callback.counter = 0

    x0 = ac_estimate.flatten()
    b = data
    al = alambda
    rl = rlambda
    W = al * A.H * D * A + rl * I + flambda * F_reg
    d = al * A.H * D * b + rl * x0
    ret, info = cg(W, d, x0=x0, maxiter=maxiter, callback=callback)
    ac = ret.reshape((M,)*3)
    assert np.all(np.isreal(ac))  # if use_reciprocal_symmetry
    ac = ac.real
    it_number = callback.counter

    print(f"Recovered AC in {it_number} iterations.", flush=True)
    image.show_volume(ac, parms.Mquat, f"autocorrelation_{generation}.png")

    return ac
