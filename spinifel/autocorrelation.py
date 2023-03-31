#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Define the Forward and Ajoint Operators"""


import numpy as np
import skopi as skp
import PyNVTX as nvtx
from spinifel import SpinifelSettings, SpinifelContexts, Profiler, settings, utils
from .extern import nufft_3d_t1, nufft_3d_t2


# ______________________________________________________________________________
# Load global settings, and contexts
#

settings = SpinifelSettings()
context = SpinifelContexts()
profiler = Profiler()
logger = utils.Logger(True, settings)


xp = np
if settings.use_cupy:
    if settings.verbose:
        logger.log(f"Using CuPy for FFTs.", level=1)
    import cupy as xp

if settings.use_fftx:
    if settings.verbose:
        print(f"Using FFTX for FFTs.")
    import fftx as fftxp
    # fftx_options_cuda = {'cuda' : True}
    # fftx_options_nocuda = {'cuda' : False}


def forward(ugrid, H_, K_, L_, support, use_recip_sym, N):
    """
    Compute the forward NUFFT: from a uniform to nonuniform set of points.

    :param ugrid: 3d array with grid sampling
    :param H_: H dimension of reciprocal space position to evaluate
    :param K_: K dimension of reciprocal space position to evaluate
    :param L_: L dimension of reciprocal space position to evaluate
    :param support: 3d object support array
    :param use_recip_sym: if True, discard imaginary component # name seems misleading
    :return nuvect: Fourier transform of uvect sampled at nonuniform (H_, K_, L_)
    """

    # make sure that points lie within permissible finufft domain
    assert np.max(np.abs(np.array([H_, K_, L_]))) < 3 * np.pi

    # Check if recip symmetry is met
    if use_recip_sym:
        assert np.all(np.isreal(ugrid))

    # Apply support if given, overwriting input array
    if support is not None:
        ugrid *= support

    # Allocate space in memory and solve NUFFT
    #nuvect = np.zeros(H_.shape, dtype=np.complex64)
    #nfft.nufft3d2(H_, K_, L_, ugrid, out=nuvect, eps=1.0e-12, isign=-1)
    start_time = time.time()
    nuvect = nufft_3d_t2(H_, K_, L_, ugrid, -1, 1e-12, N)
    end_time = time.time()
    nufft_time = end_time - start_time
    print(f"NUFFT2 time {nufft_time} on shape={H_.shape} dtype={H_.dtype} K {K_.dtype} L {L_.dtype} ugrid shape={ugrid.shape} dtype={ugrid.dtype} N={N}")

    return nuvect


@profiler.intercept
@nvtx.annotate("autocorrelation.py", is_prefix=True)
def forward_spinifel(ugrid, H_, K_, L_, support, M, N, recip_extent, use_recip_sym):
    """Apply the forward, NUFFT2- problem"""

    # Ensure that H_, K_, and L_ have the same shape
    assert H_.shape == K_.shape == L_.shape

    # Apply Support
    ugrid *= support  # /!\ overwrite

    # Apply recip symmetry
    if use_recip_sym:
        ugrid = np.fft.fftshift(
            np.fft.ifftn(np.fft.fftn(np.fft.ifftshift(ugrid.reshape((M,) * 3))).real)
        ).real

    # Solve NUFFT2-
    start_time = time.time()
    nuvect = nufft_3d_t2(H_, K_, L_, ugrid, -1, 1e-12, N)
    end_time = time.time()
    nufft_time = end_time - start_time
    print(f"NUFFT2 time {nufft_time} on shape={H_.shape} dtype={H_.dtype} K {K_.dtype} L {L_.dtype} ugrid shape={ugrid.shape} dtype={ugrid.dtype} N={N}")

    return nuvect / M**3


def adjoint(nuvect, H_, K_, L_, M, use_recip_sym=True, support=None):
    """
    Compute the adjoint NUFFT: from a nonuniform to uniform set of points.
    The sign is set for this to be the inverse FT.

    :param nuvect: flattened data vector sampled in nonuniform space
    :param H_: H dimension of reciprocal space position
    :param K_: K dimension of reciprocal space position
    :param L_: L dimension of reciprocal space position
    :param M: cubic length of desired output array
    :param support: 3d object support array
    :param use_recip_sym: if True, discard imaginary component # name seems misleading
    :return ugrid: Fourier transform of nuvect, sampled on a uniform grid
    """

    # make sure that points lie within permissible finufft domain
    assert np.max(np.abs(np.array([H_, K_, L_]))) < 3 * np.pi

    # Allocating space in memory and sovling NUFFT
    #ugrid = np.zeros((M,)*3, dtype=np.complex64)
    #nfft.nufft3d1(H_, K_, L_, nuvect, out=ugrid, eps=1.0e-15, isign=1)
    start_time = time.time()
    ugrid = nufft_3d_t1(H_, K_, L_, nuvect, 1, 1e-12, M, M, M)
    end_time = time.time()
    nufft_time = end_time - start_time
    print(f"NUFFT1 time {nufft_time} on shape={H_.shape} dtype={H_.dtype} K {K_.dtype} L {L_.dtype} nuvect shape={nuvect.shape} dtype={nuvect.dtype} M={M}")

    # Apply support if given
    if support is not None:
        ugrid *= support

    # Discard imaginary component
    if use_recip_sym:
        ugrid = ugrid.real

    return ugrid / (M**3)


@profiler.intercept
@nvtx.annotate("autocorrelation.py", is_prefix=True)
def adjoint_spinifel(nuvect, H_, K_, L_, support, M, recip_extent, use_recip_sym):
    """Apply the adjoint, NUFFT1+ problem"""

    # Ensure that H_, K_, and L_ have the same shape
    assert H_.shape == K_.shape == L_.shape

    # Solve the NUFFT
    start_time = time.time()
    ugrid = nufft_3d_t1(H_, K_, L_, nuvect, 1, 1e-12, M, M, M)
    end_time = time.time()
    nufft_time = end_time - start_time
    print(f"NUFFT1 time {nufft_time} on shape={H_.shape} dtype={H_.dtype} K {K_.dtype} L {L_.dtype} nuvect shape={nuvect.shape} dtype={nuvect.dtype} M={M}")

    # Apply support
    ugrid *= support

    # Apply recip symmetry
    if use_recip_sym:
        ugrid = np.fft.fftshift(
            np.fft.ifftn(np.fft.fftn(np.fft.ifftshift(ugrid.reshape((M,) * 3))).real)
        ).real

    return ugrid / M**3


@nvtx.annotate("autocorrelation.py", is_prefix=True)
def core_problem(
    uvect,
    H_,
    K_,
    L_,
    ac_support,
    weights,
    M,
    N,
    reciprocal_extent,
    use_reciprocal_symmetry,
):
    ugrid = uvect.reshape((M,) * 3)
    nuvect = forward(
        ugrid, H_, K_, L_, ac_support, M, N, reciprocal_extent, use_reciprocal_symmetry
    )
    nuvect *= weights
    ugrid_ADA = adjoint(
        nuvect, H_, K_, L_, ac_support, M, reciprocal_extent, use_reciprocal_symmetry
    )
    uvect_ADA = ugrid_ADA.flatten()
    return uvect_ADA


def core_problem_convolution(
    uvect, M, F_ugrid_conv_, M_ups, ac_support, use_recip_sym=True
):
    """
    Convolve data vector and input kernel of where data sample reciprocal
    space in upsampled regime.

    :param uvect: data vector on uniform grid, flattened
    :param M: length of data vector along each axis
    :param F_ugrid_conv_: Fourier transform of convolution sampling array
    :param M_ups: length of data vector along each axis when upsampled
    :param ac_support: 2d support object for autocorrelation
    :param use_recip_sym: if True, discard imaginary componeent
    :return ugrid_conv_out: convolution of uvect and F_ugrid_conv_, flattened
    """
    if use_recip_sym:
        assert np.all(np.isreal(uvect))
    # Upsample
    ugrid = uvect.reshape((M,)*3) * ac_support
    if settings.use_fftx:
        fftxp.utils.print_array_info(np, ac_support, "DATA ac_support")
        fftxp.utils.print_array_info(np, uvect, "DATA uvect")
        fftxp.utils.print_array_info(np, ugrid, "DATA ugrid")
    start_time = time.time()
    ugrid_ups = np.zeros((M_ups,)*3, dtype=uvect.dtype)
    ugrid_ups[:M, :M, :M] = ugrid
    # Convolution = Fourier multiplication
    F_ugrid_ups = np.fft.fftn(np.fft.ifftshift(ugrid_ups))
    F_ugrid_conv_out_ups = F_ugrid_ups * F_ugrid_conv_
    ugrid_conv_out_ups = np.fft.fftshift(np.fft.ifftn(F_ugrid_conv_out_ups))
    # Downsample
    ugrid_conv_out = ugrid_conv_out_ups[:M, :M, :M]
    end_time = time.time()
    np_time = end_time - start_time
    if settings.use_fftx:
        start_time = time.time()
        ugrid_conv_out_fftx = fftxp.kernels.core_problem_convolution_kernel(np, ugrid, M,  F_ugrid_conv_, M_ups)
        end_time = time.time()
        fftxp_time = end_time - start_time
        fftxp.utils.print_diff(np, ugrid_conv_out, ugrid_conv_out_fftx,
                               "core_problem_convolution ugrid_conv_out")
        print(f"FULL TIME core_problem_convolution: np {np_time} fftxp {fftxp_time}")

    ugrid_conv_out *= ac_support
    if use_recip_sym:
        # Both ugrid_conv and ugrid are real, so their convolution
        # should be real, but numerical errors accumulate in the
        # imaginary part.
        ugrid_conv_out = ugrid_conv_out.real
    return ugrid_conv_out.flatten()


@nvtx.annotate("autocorrelation.py", is_prefix=True)
def core_problem_convolution_spinifel(
    uvect, M, F_ugrid_conv_, M_ups, ac_support, use_reciprocal_symmetry
):
    if settings.use_cupy:
        uvect = xp.asarray(uvect)
        ac_support = xp.asarray(ac_support)
        F_ugrid_conv_ = xp.asarray(F_ugrid_conv_)

    # Upsample
    uvect = np.fft.fftshift(
        np.fft.ifftn(np.fft.fftn(np.fft.ifftshift(uvect.reshape((M,) * 3))).real)
    ).real
    ugrid = uvect * ac_support
    if settings.use_fftx:
        fftxp.utils.print_array_info(xp, ac_support, "DATA ac_support")
        fftxp.utils.print_array_info(xp, uvect, "DATA uvect")
        fftxp.utils.print_array_info(xp, ugrid, "DATA ugrid")
    start_time = time.time()
    ugrid_ups = xp.zeros((M_ups,) * 3, dtype=uvect.dtype)
    ugrid_ups[:M, :M, :M] = ugrid

    # Convolution = Fourier multiplication
    F_ugrid_ups = xp.fft.fftn(xp.fft.ifftshift(ugrid_ups)) / M**3 * (M_ups / M) ** 3
    F_ugrid_conv_out_ups = F_ugrid_ups * F_ugrid_conv_
    ugrid_conv_out_ups = xp.fft.fftshift(xp.fft.ifftn(F_ugrid_conv_out_ups))

    # Downsample
    ugrid_conv_out = ugrid_conv_out_ups[:M, :M, :M]
    end_time = time.time()
    xp_time = end_time - start_time
    if settings.use_fftx:
        start_time = time.time()
        # ugrid_conv_out_fftx = fftxp.convo.mdrfsconv(ugrid, F_ugrid_conv_) / M**3 * (M_ups/M)**3
        # REUSE ugrid as ugrid_conv_out_fftx.
        # Need an array of shape=(M, M, M) dtype=float64 C=True CuPy=True
        ugrid = fftxp.convo.mdrfsconv(ugrid, F_ugrid_conv_, ugrid) # REUSE ugrid as ugrid_conv_out_fftx
        ugrid *= (M_ups/M)**3 / M**3 # REUSE ugrid as ugrid_conv_out_fftx
        end_time = time.time()
        fftxp_time = end_time - start_time
        # fftxp.utils.print_diff(xp, ugrid_conv_out, ugrid_conv_out_fftx,
        fftxp.utils.print_diff(xp, ugrid_conv_out, ugrid, # REUSE ugrid as ugrid_conv_out_fftx
                               "core_problem_convolution_spinifel ugrid_conv_out")
        print(f"FULL TIME core_problem_convolution_spinifel: xp {xp_time} fftxp {fftxp_time}")
    
    ugrid_conv_out *= ac_support

    # Apply recip symmetry
    if use_reciprocal_symmetry:
        # Both ugrid_conv and ugrid are real, so their convolution
        # should be real, but numerical errors accumulate in the
        # imaginary part.
        ugrid_conv_out = np.fft.fftshift(
            np.fft.ifftn(
                np.fft.fftn(np.fft.ifftshift(ugrid_conv_out.reshape((M,) * 3))).real
            )
        ).real

    if settings.use_cupy:
        ugrid_conv_out = xp.asnumpy(ugrid_conv_out)
    return ugrid_conv_out.flatten()


@nvtx.annotate("autocorrelation.py", is_prefix=True)
def fourier_reg(uvect, support, F_antisupport, M, use_recip_sym):
    ugrid = uvect.reshape((M,) * 3) * support
    if use_recip_sym:
        assert np.all(np.isreal(ugrid))

    if settings.use_cupy:
        uvect = xp.asarray(uvect)
        support = xp.asarray(support)
        F_antisupport = xp.asarray(F_antisupport)

    start_time = time.time()
    F_ugrid = xp.fft.fftn(xp.fft.ifftshift(ugrid)) #/ M**3
    end_time = time.time()
    xp_time = end_time - start_time
    if settings.use_fftx:
        print(f"ORDER fftn xp.fft.ifftshift(ugrid) is {xp.fft.ifftshift(ugrid).flags.c_contiguous}")
        start_time = time.time()
        F_ugrid_fftx = fftxp.fft.fftn(xp.fft.ifftshift(ugrid))
        end_time = time.time()
        fftxp_time = end_time - start_time
        fftxp.utils.print_diff(xp, F_ugrid, F_ugrid_fftx,
                               "F_ugrid")
        print(f"FULL TIME F_ugrid_fftx: xp {xp_time} fftxp {fftxp_time}")
        fftxp.utils.print_array_info(xp, xp.fft.ifftshift(ugrid), "DATA shifted ugrid")
    
    F_reg = F_ugrid * xp.fft.ifftshift(F_antisupport)
    start_time = time.time()
    reg = xp.fft.fftshift(xp.fft.ifftn(F_reg))
    end_time = time.time()
    xp_time = end_time - start_time
    if settings.use_fftx:
        print(f"ORDER ifftn F_reg is {F_reg.flags.c_contiguous}")
        start_time = time.time()
        reg_fftx = xp.fft.fftshift(fftxp.fft.ifftn(F_reg))
        end_time = time.time()
        fftxp_time = end_time - start_time
        fftxp.utils.print_diff(xp, reg, reg_fftx, "reg")
        print(f"FULL TIME reg: xp {xp_time} fftxp {fftxp_time}")
        fftxp.utils.print_array_info(xp, F_reg, "DATA F_reg")

    uvect = (reg * support).flatten()
    if use_recip_sym:
        uvect = uvect.real

    if settings.use_cupy:
        uvect = xp.asnumpy(uvect)
    return uvect


@nvtx.annotate("autocorrelation.py", is_prefix=True)
def gen_nonuniform_positions(orientations, pixel_position_reciprocal):
    # Generate q points (h,k,l) from the given rotations and pixel positions

    if orientations.shape[0] > 0:
        rotmat = np.array(
            [np.linalg.inv(skp.quaternion2rot3d(quat)) for quat in orientations]
        )
    else:
        rotmat = np.zeros((0, 3, 3))
        logger.log(
            "WARNING: gen_nonuniform_positions got empty orientation - returning h,k,l for Null rotation",
            level=1
        )

    # TODO: How to ensure we support all formats of pixel_position reciprocal
    # Current support shape is (3, N_panels, Dim_x, Dim_y)
    H, K, L = np.einsum("ijk,klmn->jilmn", rotmat, pixel_position_reciprocal)
    # H, K, L = np.einsum("ijk,klm->jilm", rotmat, pixel_position_reciprocal)
    # shape -> [N_images] x det_shape
    return H, K, L


@nvtx.annotate("autocorrelation.py", is_prefix=True)
def gen_nonuniform_normalized_positions(
    orientations, pixel_position_reciprocal, reciprocal_extent, oversampling
):
    H, K, L = gen_nonuniform_positions(orientations, pixel_position_reciprocal)

    # TODO: Control/set precisions needed here
    # scale and change type for compatibility with finufft
    H_ = H.flatten() / reciprocal_extent * np.pi / oversampling
    K_ = K.flatten() / reciprocal_extent * np.pi / oversampling
    L_ = L.flatten() / reciprocal_extent * np.pi / oversampling

    return H_, K_, L_
