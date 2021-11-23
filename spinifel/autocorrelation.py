#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Define the Forward and Ajoint Operators"""



import numpy     as np
import skopi     as skp
import PyNVTX    as nvtx
from   spinifel  import SpinifelSettings, SpinifelContexts, Profiler, settings
from   .extern   import nufft_3d_t1, nufft_3d_t2


#______________________________________________________________________________
# Load global settings, and contexts
#

settings = SpinifelSettings()
context  = SpinifelContexts()
profiler = Profiler()


xp = np
if settings.use_cupy:
    if settings.verbose:
        print(f"Using CuPy for FFTs.")
    import cupy as xp



@profiler.intercept
@nvtx.annotate("autocorrelation.py", is_prefix=True)
def forward(ugrid, H_, K_, L_, support, M, N, recip_extent, use_recip_sym):
    """Apply the forward, NUFFT2- problem"""

    # Check if recip symmetry is met
    if use_recip_sym:
        assert np.all(np.isreal(ugrid))

    # Ensure that H_, K_, and L_ have the same shape
    assert H_.shape == K_.shape == L_.shape


    # Apply Support
    ugrid *= support  # /!\ overwrite

    # Solve NUFFT2-
    nuvect = nufft_3d_t2(H_, K_, L_, ugrid, -1, 1e-12, N)

    return nuvect #/ M**3



@profiler.intercept
@nvtx.annotate("autocorrelation.py", is_prefix=True)
def adjoint(nuvect, H_, K_, L_, support, M, recip_extent, use_recip_sym):
    """Apply the adjoint, NUFFT1+ problem"""

    # Ensure that H_, K_, and L_ have the same shape
    assert H_.shape == K_.shape == L_.shape

    # Solve the NUFFT
    ugrid = nufft_3d_t1(H_, K_, L_, nuvect, 1, 1e-12, M, M, M)

    # Apply support
    ugrid *= support

    # Apply recip symmetry
    if use_recip_sym:
        ugrid = ugrid.real

    return ugrid



@nvtx.annotate("autocorrelation.py", is_prefix=True)
def core_problem(uvect, H_, K_, L_, ac_support, weights, M, N,
                 reciprocal_extent, use_reciprocal_symmetry):
    ugrid = uvect.reshape((M,) * 3)
    nuvect = forward(
        ugrid, H_, K_, L_, ac_support, M, N,
        reciprocal_extent, use_reciprocal_symmetry)
    nuvect *= weights
    ugrid_ADA = adjoint(
        nuvect, H_, K_, L_, ac_support, M,
        reciprocal_extent, use_reciprocal_symmetry)
    uvect_ADA = ugrid_ADA.flatten()
    return uvect_ADA



@nvtx.annotate("autocorrelation.py", is_prefix=True)
def core_problem_convolution(uvect, M, F_ugrid_conv_, M_ups, ac_support,
                             use_reciprocal_symmetry):
    if settings.use_cupy:
        uvect = xp.asarray(uvect)
        ac_support = xp.asarray(ac_support)
        F_ugrid_conv_ = xp.asarray(F_ugrid_conv_)
    # Upsample
    ugrid = uvect.reshape((M,) * 3) * ac_support
    ugrid_ups = xp.zeros((M_ups,) * 3, dtype=uvect.dtype)            
    ugrid_ups[:M, :M, :M] = ugrid
    # Convolution = Fourier multiplication
    F_ugrid_ups = xp.fft.fftn(xp.fft.ifftshift(ugrid_ups)) #/ M**3
    F_ugrid_conv_out_ups = F_ugrid_ups * F_ugrid_conv_
    ugrid_conv_out_ups = xp.fft.fftshift(xp.fft.ifftn(F_ugrid_conv_out_ups))
    # Downsample
    ugrid_conv_out = ugrid_conv_out_ups[:M, :M, :M]

    ugrid_conv_out *= ac_support
    if use_reciprocal_symmetry:
        # Both ugrid_conv and ugrid are real, so their convolution
        # should be real, but numerical errors accumulate in the
        # imaginary part.
        ugrid_conv_out = ugrid_conv_out.real

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

    F_ugrid = xp.fft.fftn(xp.fft.ifftshift(ugrid)) #/ M**3
    F_reg = F_ugrid * xp.fft.ifftshift(F_antisupport)
    reg = xp.fft.fftshift(xp.fft.ifftn(F_reg))
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
        #rotmat = np.array([skp.quaternion2rot3d(quat) for quat in orientations])
        # TODO: we may not need to transpose the orientations if 
        # they were generated randomly.
        rotmat = np.array([np.linalg.inv(skp.quaternion2rot3d(quat)) for quat in orientations])
    else:
        rotmat = np.zeros((0, 3, 3))
        print("WARNING: gen_nonuniform_positions got empty orientation - returning h,k,l for Null rotation")

    # TODO: How to ensure we support all formats of pixel_position reciprocal
    # Current support shape is (3, N_panels, Dim_x, Dim_y) 
    H, K, L = np.einsum("ijk,klmn->jilmn", rotmat, pixel_position_reciprocal)
    #H, K, L = np.einsum("ijk,klm->jilm", rotmat, pixel_position_reciprocal)
    # shape -> [N_images] x det_shape
    return H, K, L



@nvtx.annotate("autocorrelation.py", is_prefix=True)
def gen_nonuniform_normalized_positions(orientations, pixel_position_reciprocal,
        reciprocal_extent, oversampling):
    H, K, L = gen_nonuniform_positions(orientations, pixel_position_reciprocal)

    # TODO: Control/set precisions needed here
    # scale and change type for compatibility with finufft
    H_ = H.flatten() / reciprocal_extent * np.pi / oversampling
    K_ = K.flatten() / reciprocal_extent * np.pi / oversampling
    L_ = L.flatten() / reciprocal_extent * np.pi / oversampling

    return H_, K_, L_
