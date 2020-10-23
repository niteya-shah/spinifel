import numpy     as np
import pysingfel as ps
from   spinifel  import SpinifelSettings

settings = SpinifelSettings()

#________________________________________________________________________________
# TRY to import cufinufft -- if it exists in the path. If cufinufft could be
# loaded, then the forward and adjoint functions default to the GPU version
#

import importlib
import finufftpy as nfft

CUFINUFFT_LOADER    = importlib.find_loader("cufinufft")
CUFINUFFT_AVAILABLE = CUFINUFFT_LOADER is not None

if settings.using_cuda and  CUFINUFFT_AVAILABLE:
    import pycuda.autoinit # NOQA:401
    from   pycuda.gpuarray import GPUArray, to_gpu
    from   cufinufft       import cufinufft

#-------------------------------------------------------------------------------



def forward_cpu(ugrid, H_, K_, L_, support, M, N, recip_extent, use_recip_sym):
    """Apply the forward, NUFFT2- problem -- CPU Implementation"""

    if settings.verbose:
        print("Using CPU to solve the forward transform")

    # Check if recip symmetry is met
    if use_recip_sym:
        assert np.all(np.isreal(ugrid))

    # Apply Support
    ugrid *= support  # /!\ overwrite

    # Allocate space in memory
    nuvect = np.zeros(N, dtype=np.complex)

    #___________________________________________________________________________
    # Solve the NUFFT
    #

    assert not nfft.nufft3d2(H_, K_, L_, nuvect, -1, 1e-12, ugrid)

    #
    #---------------------------------------------------------------------------

    return nuvect / M**3



def forward_gpu(ugrid, H_, K_, L_, support, M, N, recip_extent, use_recip_sym):
    """Apply the forward, NUFFT2- problem -- CUDA Implementation."""

    if settings.verbose:
        print("Using CUDA to solve the forward transform")

    # Set up data types/dim/shape of transform
    complex_dtype = np.complex128
    dtype         = np.float64
    dim           = 3
    shape         = (M, M, M)
    tol           = 1e-12

    # Ensure that H_, K_, and L_ have the same shape
    assert H_.shape == K_.shape == L_.shape

    # Apply Support
    ugrid *= support  # /!\ overwrite

    # Copy input data to Device (if not already there)
    if not isinstance(H_, GPUArray):
        H_gpu = to_gpu(H_)
        K_gpu = to_gpu(K_)
        L_gpu = to_gpu(L_)
        ugrid_gpu = to_gpu(ugrid.astype(complex_dtype))
    else:
        H_gpu = H_
        K_gpu = K_
        L_gpu = L_
        ugrid_gpu = ugrid.astype(complex_dtype)

    # Check if recip symmetry is met
    if use_recip_sym:
        assert np.all(np.isreal(ugrid))

    # Allocate space on Device
    # nuvect = np.zeros(N, dtype=np.complex)
    nuvect_gpu = GPUArray(shape=(N,), dtype=complex_dtype)

    #___________________________________________________________________________
    # Solve the NUFFT
    #

    # Change default NUFFT Behaviour
    forward_opts = cufinufft.default_opts(nufft_type=2, dim=dim)
    forward_opts.gpu_method = 1   # Override with method 1. The default is 2

    # Run NUFFT
    plan = cufinufft(2, shape, -1, tol, dtype=dtype, opts=forward_opts)
    plan.set_pts(H_.shape[0], H_gpu, K_gpu, L_gpu)
    plan.execute(nuvect_gpu, ugrid_gpu)

    #
    #---------------------------------------------------------------------------

    # Copy result back to host -- if the incoming data was on host
    if not isinstance(H_, GPUArray):
        nuvect = nuvect_gpu.get()
    else:
        nuvect = nuvect_gpu

    return nuvect / M**3



def adjoint_cpu(nuvect, H_, K_, L_, support, M, recip_extent, use_recip_sym):
    """Apply the adjoint, NUFFT1+ problem -- CPU Implementation"""

    if settings.verbose:
        print("Using CPU to solve the adjoint transform")

    # Allocating space in memory
    ugrid = np.zeros((M,)*3, dtype=np.complex, order='F')

    #___________________________________________________________________________
    # Solve the NUFFT
    #

    assert not nfft.nufft3d1(H_, K_, L_, nuvect, +1, 1e-12, M, M, M, ugrid)

    #
    #---------------------------------------------------------------------------

    # Apply support
    ugrid *= support

    # Apply recip symmetry
    if use_recip_sym:
        ugrid = ugrid.real

    return ugrid



def adjoint_gpu(nuvect, H_, K_, L_, support, M, recip_extent, use_recip_sym):
    """Apply the adjoint, NUFFT1+ problem -- CUDA Implementation."""

    if settings.verbose:
        print("Using CUDA to solve the adjoint transform")

    # Set up data types/dim/shape of transform
    complex_dtype = np.complex128
    dtype         = np.float64
    dim           = 3
    shape         = (M, M, M)
    tol           = 1e-12

    # Ensure that H_, K_, and L_ have the same shape
    assert H_.shape == K_.shape == L_.shape

    # Copy input data to Device (if not already there)
    if not isinstance(H_, GPUArray):
        H_gpu = to_gpu(H_)
        K_gpu = to_gpu(K_)
        L_gpu = to_gpu(L_)
        nuvect_gpu = to_gpu(nuvect.astype(complex_dtype))
    else:
        H_gpu = H_
        K_gpu = K_
        L_gpu = L_
        nuvect_gpu = nuvect.astype(complex_dtype)

    # Allocate space on Device
    ugrid_gpu = GPUArray(shape, dtype=complex_dtype, order="F")

    #___________________________________________________________________________
    # Solve the NUFFT
    #

    # Change default NUFFT Behaviour
    adjoint_opts = cufinufft.default_opts(nufft_type=1, dim=dim)
    adjoint_opts.gpu_method = 1   # Override with method 1. The default is 2

    # Run NUFFT
    plan = cufinufft(1, shape, 1, tol, dtype=dtype, opts=adjoint_opts)
    plan.set_pts(H_.shape[0], H_gpu, K_gpu, L_gpu)
    plan.execute(nuvect_gpu, ugrid_gpu)

    #
    #---------------------------------------------------------------------------

    # Copy result back to host -- if the incoming data was on host
    if not isinstance(H_, GPUArray):
        ugrid = ugrid_gpu.get()
    else:
        ugrid = ugrid_gpu

    # Apply support
    ugrid *= support

    # Apply recip symmetry
    if use_recip_sym:
        ugrid = ugrid.real

    return ugrid



#_______________________________________________________________________________
# Select the GPU vs the CPU version dependion on weather cufinufft is available
#

if settings.using_cuda and CUFINUFFT_AVAILABLE:
    forward = forward_gpu
    adjoint = adjoint_gpu
else:
    forward = forward_cpu
    adjoint = adjoint_cpu

#-------------------------------------------------------------------------------



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



def gen_nonuniform_positions(orientations, pixel_position_reciprocal):
    if orientations.shape[0] > 0:
        rotmat = np.array([ps.quaternion2rot3d(quat) for quat in orientations])
    else:
        rotmat = np.zeros((0, 3, 3))
    H, K, L = np.einsum("ijk,klmn->jilmn", rotmat, pixel_position_reciprocal)
    # shape -> [N_images] x det_shape
    return H, K, L
