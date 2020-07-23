import finufftpy as nfft
import numpy as np

import pysingfel as ps


def forward(ugrid, H_, K_, L_, support, M, N,
            recip_extent, use_recip_sym):
    """Apply the forward, NUFFT2- problem."""
    if use_recip_sym:
        assert np.all(np.isreal(ugrid))
    ugrid *= support  # /!\ overwrite
    nuvect = np.zeros(N, dtype=np.complex)
    assert not nfft.nufft3d2(H_, K_, L_, nuvect, -1, 1e-12, ugrid)
    return nuvect / M**3


def adjoint(nuvect, H_, K_, L_, support, M,
            recip_extent, use_recip_sym):
    """Apply the adjoint, NUFFT1+ problem."""
    ugrid = np.zeros((M,)*3, dtype=np.complex, order='F')
    assert not nfft.nufft3d1(H_, K_, L_, nuvect, +1, 1e-12, M, M, M, ugrid)
    ugrid *= support
    if use_recip_sym:
        ugrid = ugrid.real
    return ugrid


def core_problem(uvect, H_, K_, L_, ac_support, weights, M, N,
                 reciprocal_extent, use_reciprocal_symmetry):
    ugrid = uvect.reshape((M,)*3)
    nuvect = forward(
        ugrid, H_, K_, L_, ac_support, M, N,
        reciprocal_extent, use_reciprocal_symmetry)
    nuvect *= weights
    ugrid_ADA = adjoint(
        nuvect, H_, K_, L_, ac_support, M,
        reciprocal_extent, use_reciprocal_symmetry)
    uvect_ADA = ugrid_ADA.flatten()
    return uvect_ADA


def core_problem_convolution(uvect, M, F_ugrid_conv_, M_ups, ac_support,
                             use_reciprocal_symmetry):
    if use_reciprocal_symmetry:
        assert np.all(np.isreal(uvect))
    # Upsample
    ugrid = uvect.reshape((M,)*3) * ac_support
    ugrid_ups = np.zeros((M_ups,)*3, dtype=uvect.dtype)
    ugrid_ups[:M, :M, :M] = ugrid
    # Convolution = Fourier multiplication
    F_ugrid_ups = np.fft.fftn(np.fft.ifftshift(ugrid_ups))
    F_ugrid_conv_out_ups = F_ugrid_ups * F_ugrid_conv_
    ugrid_conv_out_ups = np.fft.fftshift(np.fft.ifftn(F_ugrid_conv_out_ups))
    # Downsample
    ugrid_conv_out = ugrid_conv_out_ups[:M, :M, :M]
    ugrid_conv_out *= ac_support
    if use_reciprocal_symmetry:
        # Both ugrid_conv and ugrid are real, so their convolution
        # should be real, but numerical errors accumulate in the
        # imaginary part.
        ugrid_conv_out = ugrid_conv_out.real
    return ugrid_conv_out.flatten()


def gen_nonuniform_positions(orientations, pixel_position_reciprocal):
    rotmat = np.array([ps.quaternion2rot3d(quat) for quat in orientations])
    H, K, L = np.einsum("ijk,klmn->jilmn", rotmat, pixel_position_reciprocal)
    # shape -> [N_images] x det_shape
    return H, K, L