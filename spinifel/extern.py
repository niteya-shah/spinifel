#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Manages external libraries"""


from   logging            import getLogger
from   sys                import getsizeof
from   importlib.metadata import version
import numpy    as np
import PyNVTX   as nvtx
from   spinifel import SpinifelSettings, SpinifelContexts, Profiler



#______________________________________________________________________________
# Load global settings, and contexts
#

settings = SpinifelSettings()
context  = SpinifelContexts()
profiler = Profiler()



#______________________________________________________________________________
# Load cufiNUFFT or fiNUFFTpy depending on settings: use_cuda, use_cufinufft
#

class CUFINUFFTRequiredButNotFound(Exception):
    """Settings require cufiNUFFT, but the module is unavailable"""



class CUFINUFFTVersionUnsupported(Exception):
    """The detected version of cufiNUFFT, is unsupported"""



class FINUFFTPYRequiredButNotFound(Exception):
    """Settings require cufiNUFFT, but the module is unavailable"""



class FINUFFTPYVersionUnsupported(Exception):
    """The detected version of fiNUFFT, is unsupported"""




if settings.using_cuda and settings.use_cufinufft:
    # TODO: only manage MPI via contexts! But let's leave this here for now
    context.init_mpi()  # Ensures that MPI has been initalized
    context.init_cuda() # this must be called _after_ init_mpi
    from pycuda.gpuarray import GPUArray, to_gpu

    if context.cufinufft_available:
        from cufinufft import cufinufft
        FINUFFT_CUDA = True
    else:
        raise CUFINUFFTRequiredButNotFound
else:
    if context.finufftpy_available:
        import finufftpy as nfft
        FINUFFT_CUDA = False
    else:
        raise FINUFFTPYRequiredButNotFound



@profiler.intercept
def nufft_3d_t1_finufft_v1(x, y, z, nuvect, sign, eps, nx, ny, nz):
    """
    Version 1 of fiNUFFT 3D type 1
    """

    if settings.verbose:
        print("Using CPU to solve the NUFFT 3D T1")

    # Ensure that x, y, and z have the same shape
    assert x.shape == y.shape == z.shape

    # Allocating space in memory
    ugrid = np.zeros((nx, ny, nz), dtype=np.complex, order='F')

    #__________________________________________________________________________
    # Solve the NUFFT
    #

    assert not nfft.nufft3d1(x, y, z, nuvect, sign, eps, nx, ny, nz, ugrid)

    #
    #--------------------------------------------------------------------------

    return ugrid



@profiler.intercept
def nufft_3d_t2_finufft_v1(x, y, z, ugrid, sign, eps, n):
    """
    Version 1 of fiNUFFT 3D type 2
    """

    if settings.verbose:
        print("Using CPU to solve the NUFFT 3D T2")

    # Ensure that x, y, and z have the same shape
    assert x.shape == y.shape == z.shape

    # Allocate space in memory
    nuvect = np.zeros(n, dtype=np.complex)

    #__________________________________________________________________________
    # Solve the NUFFT
    #

    assert not nfft.nufft3d2(x, y, z, nuvect, sign, eps, ugrid)

    #
    #--------------------------------------------------------------------------


    return nuvect



@nvtx.annotate("extern.transpose")
def transpose(x, y, z):
    """Transposes the order of the (x, y, z) coordinates to (z, y, x)"""
    return z, y, x



@profiler.intercept
@nvtx.annotate("extern.nufft_3d_t2_cufinufft_v1")
def nufft_3d_t1_cufinufft_v1(H_, K_, L_, nuvect, sign, eps, nx, ny, nz):
    """
    Version 1 of cufiNUFFT 3D type 1
    """

    dim    = 3
    shape  = (nx, ny, nz)
    dev_id = context.dev_id
    logger = getLogger(__name__)
    complex_dtype = np.complex128
    dtype         = np.float64

    if settings.verbose:
        print(f"Using CUDA to solve the NUFFT 3D T1 on device {dev_id}")

    # Ensure that H_, K_, and L_ have the same shape
    assert H_.shape == K_.shape == L_.shape

    # The rest of Spinifel uses Fortran coordinate order:
    H_, K_, L_ = transpose(H_, K_, L_)

    # Copy input data to Device (if not already there)
    nvtx.RangePushA("extren.nufft_3d_t1_cufinufft_v1:to_gpu")
    if not isinstance(H_, GPUArray):
        logger.debug(
                (
                    f"L={L_.shape}/{L_.dtype}/{getsizeof(L_)} ",
                    f"K={K_.shape}/{K_.dtype}/{getsizeof(K_)} ",
                    f"H={H_.shape}/{H_.dtype}/{getsizeof(H_)}"
                )
            )
        H_gpu = to_gpu(H_)
        K_gpu = to_gpu(K_)
        L_gpu = to_gpu(L_)
        nuvect_gpu = to_gpu(nuvect.astype(complex_dtype))
    else:
        H_gpu = H_
        K_gpu = K_
        L_gpu = L_
        nuvect_gpu = nuvect.astype(complex_dtype)
    nvtx.RangePop()

    # Allocate space on Device
    nvtx.RangePushA("extern.nufft_3d_t1_cufinufft_v1:GPUArray")
    ugrid_gpu = GPUArray(shape, dtype=complex_dtype, order="F")
    nvtx.RangePop()


    #__________________________________________________________________________
    # Solve the NUFFT
    #
    nvtx.RangePushA("autocorrelation.adjoint_gpu:setup_cufinufft")

    # Change default NUFFT Behaviour
    adjoint_opts = cufinufft.default_opts(nufft_type=1, dim=dim)
    adjoint_opts.gpu_method = 1   # Override with method 1. The default is 2
    adjoint_opts.cuda_device_id = dev_id

    # Run NUFFT
    # TODO: MONA check here performance dependent on data?
    plan = cufinufft(1, shape, sign, eps, dtype=dtype, opts=adjoint_opts)
    plan.set_pts(H_.shape[0], H_gpu, K_gpu, L_gpu)
    plan.execute(nuvect_gpu, ugrid_gpu)
    nvtx.RangePop()

    #
    #--------------------------------------------------------------------------


    # Copy result back to host -- if the incoming data was on host
    nvtx.RangePushA("autocorrelation.adjoint_gpu:get")
    if not isinstance(H_, GPUArray):
        ugrid = ugrid_gpu.get()
    else:
        ugrid = ugrid_gpu
    nvtx.RangePop()


    return ugrid



@profiler.intercept
@nvtx.annotate("extern.nufft_3d_t2_cufinufft_v1")
def nufft_3d_t2_cufinufft_v1(H_, K_, L_, ugrid, sign, eps, N):
    """
    Version 1 of cufiNUFFT 3D type 2
    """

    dim    = 3
    dev_id = context.dev_id
    logger = getLogger(__name__)
    complex_dtype = np.complex128
    dtype         = np.float64

    if settings.verbose:
        print(f"Using CUDA to solve the NUFFT 3D T2 on device {dev_id}")

    gpu_free, gpu_total = context.cuda_mem_info()
    logger.debug(
            (
                f"init gpu_free={gpu_free/1e9:.2f}GB ",
                f"gpu_total={gpu_total/1e9:.2f}GB"
            )
        )

    # Ensure that H_, K_, and L_ have the same shape
    assert H_.shape == K_.shape == L_.shape

    # The rest of Spinifel uses Fortran coordinate order:
    H_, K_, L_ = transpose(H_, K_, L_)

    # Copy input data to Device (if not already there)
    nvtx.RangePushA("extern.nufft_3d_t2_cufinufft_v1:to_gpu")
    if not isinstance(H_, GPUArray):
        H_gpu = to_gpu(H_)
        gpu_free, gpu_total = context.cuda_mem_info()
        logger.debug(
                (
                    f"H={getsizeof(H_)/1e9:.2f}GB copied ",
                    f"gpu_free={gpu_free/1e9:.2f}GB ",
                    f"gpu_total={gpu_total/1e9:.2f}GB"
                )
            )

        K_gpu = to_gpu(K_)
        gpu_free, gpu_total = context.cuda_mem_info()
        logger.debug(
                (
                    f"K={getsizeof(K_)/1e9:.2f}GB copied ",
                    f"gpu_free={gpu_free/1e9:.2f}GB ",
                    f"gpu_total={gpu_total/1e9:.2f}GB"
                )
            )

        L_gpu = to_gpu(L_)
        gpu_free, gpu_total = context.cuda_mem_info()
        logger.debug(
                (
                    f"L={getsizeof(L_)/1e9:.2f}GB copied ",
                    f"gpu_free={gpu_free/1e9:.2f}GB ",
                    f"gpu_total={gpu_total/1e9:.2f}GB"
                )
            )
        ugrid_gpu = to_gpu(ugrid.astype(complex_dtype))
    else:
        H_gpu = H_
        K_gpu = K_
        L_gpu = L_
        ugrid_gpu = ugrid.astype(complex_dtype)
    nvtx.RangePop()

    # Allocate space on Device
    nvtx.RangePushA("extern.nufft_3d_t2_cufinufft_v1:GPUArray")

    nuvect_gpu = GPUArray(shape=(N,), dtype=complex_dtype)
    gpu_free, gpu_total = context.cuda_mem_info()
    logger.debug(
            (
                f"nuvect_gpu={getsizeof(nuvect_gpu)/1e9:.2f}GB ",
                f"allocated gpu_free={gpu_free/1e9:.2f}GB ",
                f"gpu_total={gpu_total/1e9:.2f}GB"
            )
        )
    nvtx.RangePop()

    #__________________________________________________________________________
    # Solve the NUFFT
    #
    nvtx.RangePushA("extern.nufft_3d_t2_cufinufft_v1:setup_cufinufft")

    # Change default NUFFT Behaviour
    forward_opts = cufinufft.default_opts(nufft_type=2, dim=dim)
    forward_opts.gpu_method = 1   # Override with method 1. The default is 2
    forward_opts.cuda_device_id = dev_id

    # Run NUFFT
    plan = cufinufft(2, ugrid.shape, sign, eps, dtype=dtype, opts=forward_opts)
    plan.set_pts(H_.shape[0], H_gpu, K_gpu, L_gpu)
    plan.execute(nuvect_gpu, ugrid_gpu)
    nvtx.RangePop()

    #
    #--------------------------------------------------------------------------

    # Copy result back to host -- if the incoming data was on host
    nvtx.RangePushA("autocorrelation.forward_gpu:get")
    if not isinstance(H_, GPUArray):
        nuvect = nuvect_gpu.get()
    else:
        nuvect = nuvect_gpu
    nvtx.RangePop()

    return nuvect



#______________________________________________________________________________
# Alias the nufft functions to their cpu/gpu implementations
#

if settings.using_cuda and settings.use_cufinufft:
    print("Orientation Matching: USING_CUDA")

    if context.cufinufft_available:
        print("++++++++++++++++++++: USING_CUFINUFFT")
        if version("cufinufft") == "1.1":
            nufft_3d_t1 = nufft_3d_t1_cufinufft_v1
            nufft_3d_t2 = nufft_3d_t2_cufinufft_v1
        else:
            raise CUFINUFFTVersionUnsupported
    else:
        raise CUFINUFFTRequiredButNotFound
else:
    if context.finufftpy_available:
        print("++++++++++++++++++++: USING_FINUFFTPY")
        if version("finufftpy") == "1.1.2":
            nufft_3d_t1 = nufft_3d_t1_finufft_v1
            nufft_3d_t2 = nufft_3d_t2_finufft_v1
        else:
            raise FINUFFTPYVersionUnsupported
    else:
        raise FINUFFTPYRequiredButNotFound
