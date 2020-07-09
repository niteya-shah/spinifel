import finufftpy as nfft
import numpy as np

import pysingfel as ps


def forward(uvect, H_, K_, L_, support, M, N,
            recip_extent, use_recip_sym):
    """Apply the forward, NUFFT2- problem."""
    if use_recip_sym:
        assert np.all(np.isreal(uvect))
    ugrid = uvect.reshape((M,)*3) * support
    nuvect = np.zeros(N, dtype=np.complex)
    assert not nfft.nufft3d2(H_, K_, L_, nuvect, -1, 1e-12, ugrid)
    return nuvect / M**3


def adjoint(nuvect, H_, K_, L_, support, M,
            recip_extent, use_recip_sym):
    """Apply the adjoint, NUFFT1+ problem."""
    ugrid = np.zeros((M,)*3, dtype=np.complex, order='F')
    assert not nfft.nufft3d1(H_, K_, L_, nuvect, +1, 1e-12, M, M, M, ugrid)
    uvect = (ugrid * support).flatten()
    if use_recip_sym:
        uvect = uvect.real
    return uvect


def core_problem(uvect, H_, K_, L_, ac_support, weights, M, N,
                 reciprocal_extent, use_reciprocal_symmetry):
    nuvect = forward(
        uvect, H_, K_, L_, ac_support, M, N,
        reciprocal_extent, use_reciprocal_symmetry)
    nuvect *= weights
    uvect_ADA = adjoint(
        nuvect, H_, K_, L_, ac_support, M,
        reciprocal_extent, use_reciprocal_symmetry)
    return uvect_ADA


def gen_nonuniform_positions(orientations, pixel_position_reciprocal):
    rotmat = np.array([ps.quaternion2rot3d(quat) for quat in orientations])
    H, K, L = np.einsum("ijk,klmn->jilmn", rotmat, pixel_position_reciprocal)
    # shape -> [N_images] x det_shape
    return H, K, L
