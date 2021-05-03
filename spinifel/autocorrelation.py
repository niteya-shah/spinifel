import numpy     as np
import skopi     as skp
from   spinifel  import SpinifelSettings, SpinifelContexts, Profiler
import time
import logging

settings = SpinifelSettings()
context  = SpinifelContexts()
profiler = Profiler()

import PyNVTX as nvtx

import sys


import pycuda.driver as cuda

import os
USE_ORIGINAL_FINUFFT = int(os.environ.get('USE_ORIGINAL_FINUFFT', '0'))
if USE_ORIGINAL_FINUFFT:
    import finufft as nfft_original

#________________________________________________________________________________
# Load cufiNUFFT or fiNUFFTpy depending on settings: use_cuda, use_cufinufft
#

class CUFINUFFTRequiredButNotFound(Exception):
    pass



class FINUFFTPYRequiredButNotFound(Exception):
    pass



if settings.using_cuda and settings.use_cufinufft:
    print("Orientation Matching: USING_CUDA")
    # HACK: only manage MPI via contexts! But let's leave this here for now
    context.init_mpi()  # Ensures that MPI has been initalized
    context.init_cuda() # this must be called _after_ init_mpi
    from pycuda.gpuarray import GPUArray, to_gpu

    if context.cufinufft_available:
        print("++++++++++++++++++++: USING_CUFINUFFT")
        from cufinufft import cufinufft
    else:
        raise CUFINUFFTRequiredButNotFound
else:
    if context.finufftpy_available:
        import finufftpy as nfft
    else:
        raise FINUFFTPYRequiredButNotFound


#-------------------------------------------------------------------------------



@profiler.intercept
def forward_cpu(ugrid, H_, K_, L_, support, M, N, recip_extent, use_recip_sym):
    """Apply the forward, NUFFT2- problem -- CPU Implementation"""
    # Note than M, N, recip_extent are not used here - only to 
    # match the function inputs with forward_gpu
    
    if settings.verbose:
        print("Using CPU to solve the forward transform")

    # Check if recip symmetry is met
    if use_recip_sym:
        assert np.all(np.isreal(ugrid))

    # Apply Support
    ugrid *= support  # /!\ overwrite

    # TODO: Current switched to finufft interface and single precision
    # Ask Elliott if we can use this main finufft instead of his fork.
    # Allocate space in memory
    if USE_ORIGINAL_FINUFFT:
        nuvect = np.zeros(H_.shape, dtype=np.complex64)
        nfft_original.nufft3d2(H_, K_, L_, ugrid, out=nuvect, eps=6e-08, isign=-1)
        return nuvect
    
    print(f'DEBUG forward_cpu')
    print(f'ugrid={ugrid.shape} {ugrid.dtype}')
    print(f'H_={H_.shape} {H_.dtype}')
    print(f'support={support}')
    print(f'M={M} N={N} recip_extent={recip_extent} use_recip_sym={use_recip_sym}')
    nuvect = np.zeros(N, dtype=np.complex)

    #___________________________________________________________________________
    # Solve the NUFFT
    #

    assert not nfft.nufft3d2(H_, K_, L_, nuvect, -1, 1e-12, ugrid)

    #
    #---------------------------------------------------------------------------

    return nuvect / M**3
    #return nuvect



@profiler.intercept
@nvtx.annotate("autocorrelation.forward_gpu")
def forward_gpu(ugrid, H_, K_, L_, support, M, N, recip_extent, use_recip_sym):
    """Apply the forward, NUFFT2- problem -- CUDA Implementation."""
    logger = logging.getLogger(__name__)

    gpu_free, gpu_total = cuda.mem_get_info()
    logger.debug(f'init gpu_free={gpu_free/1e9:.2f}GB gpu_total={gpu_total/1e9:.2f}GB')
    
    dev_id = context.dev_id
    if settings.verbose:
        print(f"Using CUDA to solve the forward transform on device {dev_id}")

    # Set up data types/dim/shape of transform
    complex_dtype = np.complex128
    dtype         = np.float64
    dim           = 3
    shape         = (M, M, M)
    tol           = 6e-08

    # Ensure that H_, K_, and L_ have the same shape
    assert H_.shape == K_.shape == L_.shape

    # TODO 
    H_ = H_.astype(dtype)
    K_ = K_.astype(dtype)
    L_ = L_.astype(dtype)


    # Apply Support
    ugrid *= support  # /!\ overwrite

    # Copy input data to Device (if not already there)
    nvtx.RangePushA("autocorrelation.forward_gpu:to_gpu")
    if not isinstance(H_, GPUArray):
        # Due to a change to the cufinufft API, these need to be re-ordered
        # TODO: Restore when cpu finufft version has been updated
        # H_gpu = to_gpu(H_)
        # K_gpu = to_gpu(K_)
        # L_gpu = to_gpu(L_)
        
        H_gpu = to_gpu(L_)
        gpu_free, gpu_total = cuda.mem_get_info()
        logger.debug(f'H={sys.getsizeof(L_)/1e9:.2f}GB copied gpu_free={gpu_free/1e9:.2f}GB gpu_total={gpu_total/1e9:.2f}GB')
        
        K_gpu = to_gpu(K_)
        gpu_free, gpu_total = cuda.mem_get_info()
        logger.debug(f'K={sys.getsizeof(K_)/1e9:.2f}GB copied gpu_free={gpu_free/1e9:.2f}GB gpu_total={gpu_total/1e9:.2f}GB')
        
        L_gpu = to_gpu(H_)
        gpu_free, gpu_total = cuda.mem_get_info()
        logger.debug(f'L={sys.getsizeof(H_)/1e9:.2f}GB copied gpu_free={gpu_free/1e9:.2f}GB gpu_total={gpu_total/1e9:.2f}GB')
        ugrid_gpu = to_gpu(ugrid.astype(complex_dtype))
    else:
        # Due to a change to the cufinufft API, these need to be re-ordered
        # TODO: Restore when cpu finufft version has been updated
        # H_gpu = H_
        # K_gpu = K_
        # L_gpu = L_
        H_gpu = L_
        K_gpu = K_
        L_gpu = H_
        ugrid_gpu = ugrid.astype(complex_dtype)
    nvtx.RangePop()

    # Check if recip symmetry is met
    if use_recip_sym:
        assert np.all(np.isreal(ugrid))

    # Allocate space on Device
    # nuvect = np.zeros(N, dtype=np.complex)
    nvtx.RangePushA("autocorrelation.forward_gpu:GPUArray")

    
    nuvect_gpu = GPUArray(shape=(N,), dtype=complex_dtype)
    gpu_free, gpu_total = cuda.mem_get_info()
    logger.debug(f'nuvect_gpu={sys.getsizeof(nuvect_gpu)/1e9:.2f}GB allocated gpu_free={gpu_free/1e9:.2f}GB gpu_total={gpu_total/1e9:.2f}GB')
    nvtx.RangePop()


    # TODO: It looks to me like cufinufft doesn'work the dimension is increased
    # from 81 to 151. Ask Johannes on this. This applies for both forward and
    # adjoint.

    #___________________________________________________________________________
    # Solve the NUFFT
    #
    nvtx.RangePushA("autocorrelation.forward_gpu:setup_cufinufft")

    # Change default NUFFT Behaviour
    forward_opts = cufinufft.default_opts(nufft_type=2, dim=dim)
    forward_opts.gpu_method = 1   # Override with method 1. The default is 2
    forward_opts.cuda_device_id = dev_id

    # Run NUFFT
    plan = cufinufft(2, shape, -1, tol, dtype=dtype, opts=forward_opts)
    plan.set_pts(H_.shape[0], H_gpu, K_gpu, L_gpu)
    plan.execute(nuvect_gpu, ugrid_gpu)
    nvtx.RangePop()

    #
    #---------------------------------------------------------------------------

    # Copy result back to host -- if the incoming data was on host
    nvtx.RangePushA("autocorrelation.forward_gpu:get")
    if not isinstance(H_, GPUArray):
        nuvect = nuvect_gpu.get()
    else:
        nuvect = nuvect_gpu
    nvtx.RangePop()

    return nuvect / M**3



@profiler.intercept
def adjoint_cpu(nuvect, H_, K_, L_, support, M, recip_extent, use_recip_sym):
    """Apply the adjoint, NUFFT1+ problem -- CPU Implementation"""

    if settings.verbose:
        print("Using CPU to solve the adjoint transform")

    # Allocating space in memory
    ugrid = np.zeros((M,)*3, dtype=np.complex128, order='F')

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



@profiler.intercept
@nvtx.annotate("autocorrelation.adjoint_gpu")
def adjoint_gpu(nuvect, H_, K_, L_, support, M, recip_extent, use_recip_sym):
    """Apply the adjoint, NUFFT1+ problem -- CUDA Implementation."""
    logger = logging.getLogger(__name__)

    dev_id = context.dev_id
    if settings.verbose:
        print(f"Using CUDA to solve the adjoint transform on device {dev_id}")

    # Set up data types/dim/shape of transform
    complex_dtype = np.complex64
    dtype         = np.float32
    dim           = 3
    shape         = (M, M, M)
    tol           = 1e-12

    # Ensure that H_, K_, and L_ have the same shape
    assert H_.shape == K_.shape == L_.shape
    
    # Copy input data to Device (if not already there)
    nvtx.RangePushA("autocorrelation.adjoint_gpu:to_gpu")
    if not isinstance(H_, GPUArray):
        # Due to a change to the cufinufft API, these need to be re-ordered
        # TODO: Restore when cpu finufft version has been updated
        # H_gpu = to_gpu(H_)
        # K_gpu = to_gpu(K_)
        # L_gpu = to_gpu(L_)
        logging.debug(f'L={L_.shape}/{L_.dtype}/{sys.getsizeof(L_)} K={K_.shape}/{K_.dtype}/{sys.getsizeof(K_)} H={H_.shape}/{H_.dtype}/{sys.getsizeof(H_)}')
        H_gpu = to_gpu(L_)
        K_gpu = to_gpu(K_)
        L_gpu = to_gpu(H_)
        nuvect_gpu = to_gpu(nuvect.astype(complex_dtype))
    else:
        # Due to a change to the cufinufft API, these need to be re-ordered
        # TODO: Restore when cpu finufft version has been updated
        # H_gpu = H_
        # K_gpu = K_
        # L_gpu = L_
        H_gpu = L_
        K_gpu = K_
        L_gpu = H_
        nuvect_gpu = nuvect.astype(complex_dtype)
    nvtx.RangePop()

    # Allocate space on Device
    nvtx.RangePushA("autocorrelation.adjoint_gpu:GPUArray")
    ugrid_gpu = GPUArray(shape, dtype=complex_dtype, order="F")
    nvtx.RangePop()


    #___________________________________________________________________________
    # Solve the NUFFT
    #
    nvtx.RangePushA("autocorrelation.adjoint_gpu:setup_cufinufft")

    # Change default NUFFT Behaviour
    adjoint_opts = cufinufft.default_opts(nufft_type=1, dim=dim)
    adjoint_opts.gpu_method = 1   # Override with method 1. The default is 2
    adjoint_opts.cuda_device_id = dev_id
    
    # Run NUFFT
    plan = cufinufft(1, shape, 1, tol, dtype=dtype, opts=adjoint_opts) # TODO: MONA check here performance dependent on data?
    plan.set_pts(H_.shape[0], H_gpu, K_gpu, L_gpu)
    plan.execute(nuvect_gpu, ugrid_gpu)
    nvtx.RangePop()

    #
    #---------------------------------------------------------------------------


    # Copy result back to host -- if the incoming data was on host
    nvtx.RangePushA("autocorrelation.adjoint_gpu:get")
    if not isinstance(H_, GPUArray):
        ugrid = ugrid_gpu.get()
    else:
        ugrid = ugrid_gpu
    nvtx.RangePop()

    # Apply support
    ugrid *= support

    # Apply recip symmetry
    if use_recip_sym:
        ugrid = ugrid.real

    return ugrid



#_______________________________________________________________________________
# Select the GPU vs the CPU version dependion on weather cufinufft is available
#

if settings.using_cuda and settings.use_cufinufft:
    if context.cufinufft_available:
        forward = forward_gpu
        adjoint = adjoint_gpu
    else:
        raise CUFINUFFTRequiredButNotFound
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
    # Generate q points (h,k,l) from the given rotations and pixel positions 

    if orientations.shape[0] > 0:
        #rotmat = np.array([skp.quaternion2rot3d(quat) for quat in orientations])
        # TODO: we may not need to transpose the orientations if 
        # they were generated randomly.
        rotmat = np.array([np.linalg.inv(skp.quaternion2rot3d(quat)) for quat in orientations])
    else:
        rotmat = np.zeros((0, 3, 3))
        print(f"WARNING: gen_nonuniform_positions got empty orientation - returning h,k,l for Null rotation")

    # TODO: How to ensure we support all formats of pixel_position reciprocal
    # Current support shape is (3, N_panels, Dim_x, Dim_y) 
    H, K, L = np.einsum("ijk,klmn->jilmn", rotmat, pixel_position_reciprocal)
    #H, K, L = np.einsum("ijk,klm->jilm", rotmat, pixel_position_reciprocal)
    # shape -> [N_images] x det_shape
    return H, K, L

def gen_nonuniform_normalized_positions(orientations, pixel_position_reciprocal, 
        reciprocal_extent, oversampling):
    H, K, L = gen_nonuniform_positions(orientations, pixel_position_reciprocal)
    
    # TODO: Control/set precisions needed here
    # scale and change type for compatibility with finufft
    H_ = H.astype(np.float32).flatten() / reciprocal_extent * np.pi / oversampling
    K_ = K.astype(np.float32).flatten() / reciprocal_extent * np.pi / oversampling
    L_ = L.astype(np.float32).flatten() / reciprocal_extent * np.pi / oversampling

    return H_, K_, L_

