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

    # Ensure that H_, K_, and L_ have the same shape
    assert H_.shape == K_.shape == L_.shape

    # Apply Support
    ugrid *= support  # /!\ overwrite

    # Apply recip symmetry
    if use_recip_sym:
        ugrid = np.fft.fftshift(np.fft.ifftn(np.fft.fftn(np.fft.ifftshift(ugrid.reshape((M,)*3))).real)).real

    # Solve NUFFT2-
    nuvect = nufft_3d_t2(H_, K_, L_, ugrid, -1, 1e-12, N)

    return nuvect / M**3

def forward_cmtip(ugrid, H_, K_, L_, support, use_recip_sym):
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
    assert np.max(np.abs(np.array([H_, K_, L_]))) < 3*np.pi

    # Check if recip symmetry is met
    if use_recip_sym:
        assert np.all(np.isreal(ugrid))

    # Apply support if given, overwriting input array
    if support is not None:
        ugrid *= support 
        
    # Allocate space in memory and solve NUFFT
    #nuvect = np.zeros(H_.shape, dtype=np.complex64)
    nuvect = nufft_3d_t2(H_, K_, L_, ugrid, -1, 1e-12, H_.shape)
    
    return nuvect 

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
        #ugrid = np.fft.fftshift(np.fft.ifftn(np.fft.fftn(np.fft.ifftshift(ugrid.reshape((M,)*3))).real)).real
        ugrid = ugrid.real

    return ugrid / (M**3)



@nvtx.annotate("autocorrelation.py", is_prefix=True)
def core_problem(uvect, H_, K_, L_, ac_support, weights, M, N,
                 reciprocal_extent, use_reciprocal_symmetry):
    ugrid = uvect.reshape((M,) * 3)
    nuvect = forward(
        ugrid, H_, K_, L_, ac_support, M, N,
        reciprocal_extent, use_reciprocal_symmetry)
    #nuvect = forward_cmtip(
    #    ugrid, H_, K_, L_, ac_support, use_reciprocal_symmetry)
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
    uvect = np.fft.fftshift(np.fft.ifftn(np.fft.fftn(np.fft.ifftshift(uvect.reshape((M,)*3))).real)).real
    ugrid = uvect * ac_support
    ugrid_ups = xp.zeros((M_ups,) * 3, dtype=uvect.dtype)            
    ugrid_ups[:M, :M, :M] = ugrid
    
    # Convolution = Fourier multiplication
    F_ugrid_ups = xp.fft.fftn(xp.fft.ifftshift(ugrid_ups)) / M**3 * (M_ups/M)**3
    F_ugrid_conv_out_ups = F_ugrid_ups * F_ugrid_conv_
    ugrid_conv_out_ups = xp.fft.fftshift(xp.fft.ifftn(F_ugrid_conv_out_ups))
    
    # Downsample
    ugrid_conv_out = ugrid_conv_out_ups[:M, :M, :M]
    ugrid_conv_out *= ac_support

    # Apply recip symmetry
    if use_reciprocal_symmetry:
        # Both ugrid_conv and ugrid are real, so their convolution
        # should be real, but numerical errors accumulate in the
        # imaginary part.
        ugrid_conv_out = np.fft.fftshift(np.fft.ifftn(np.fft.fftn(np.fft.ifftshift(ugrid_conv_out.reshape((M,)*3))).real)).real

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
