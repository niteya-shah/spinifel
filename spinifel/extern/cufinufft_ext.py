#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Manages external libraries"""


from   logging  import getLogger
from   sys      import getsizeof
import numpy    as np
import PyNVTX   as nvtx
from   spinifel import SpinifelSettings, SpinifelContexts, Profiler
from   .        import transpose, CUFINUFFTRequiredButNotFound, \
                       FINUFFTPYRequiredButNotFound



#______________________________________________________________________________
# Load global settings, and contexts
#

settings = SpinifelSettings()
context  = SpinifelContexts()
profiler = Profiler()



#______________________________________________________________________________
# Load cufiNUFFT or fiNUFFTpy depending on settings: use_cuda, use_cufinufft
#

if settings.using_cuda and settings.use_cufinufft:
    from . import GPUArray, to_gpu

    if context.cufinufft_available:
        from . import cufinufft
    else:
        raise CUFINUFFTRequiredButNotFound



@profiler.intercept
@nvtx.annotate("extern.nufft_3d_t1_cufinufft_v1")
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
    nvtx.RangePushA("extern.nufft_3d_t1_cufinufft_v1:setup_cufinufft")

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
    nvtx.RangePushA("extern.nufft_3d_t1_cufinufft_v1:get")
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
    nvtx.RangePushA("extern.nufft_3d_t2_cufinufft_v1:get")
    if not isinstance(H_, GPUArray):
        nuvect = nuvect_gpu.get()
    else:
        nuvect = nuvect_gpu
    nvtx.RangePop()

    return nuvect




@profiler.intercept
@nvtx.annotate("extern.nufft_3d_t1_cufinufft_v2")
def nufft_3d_t1_cufinufft_v2(H_, K_, L_, nuvect, sign, eps, nx, ny, nz):
    """
    Version 2 of cufiNUFFT 3D type 1
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
    nvtx.RangePushA("extren.nufft_3d_t1_cufinufft_v2:to_gpu")
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
    nvtx.RangePushA("extern.nufft_3d_t1_cufinufft_v2:GPUArray")
    ugrid_gpu = GPUArray(shape, dtype=complex_dtype, order="F")
    nvtx.RangePop()

    #__________________________________________________________________________
    # Solve the NUFFT
    #
    nvtx.RangePushA("extern.nufft_3d_t1_cufinufft_v2:setup_cufinufft")
    # TODO: MONA check here performance dependent on data?
    plan = cufinufft(
            1, shape, 1, eps, isign=1, dtype=dtype,
            gpu_method=1, gpu_device_id=dev_id
        )
    plan.set_pts(H_gpu, K_gpu, L_gpu)
    plan.execute(nuvect_gpu, ugrid_gpu)
    nvtx.RangePop()
    #
    #--------------------------------------------------------------------------

    # Copy result back to host -- if the incoming data was on host
    nvtx.RangePushA("extern.nufft_3d_t1_cufinufft_v2:get")
    if not isinstance(H_, GPUArray):
        ugrid = ugrid_gpu.get()
    else:
        ugrid = ugrid_gpu
    nvtx.RangePop()


    return ugrid



@profiler.intercept
@nvtx.annotate("extern.nufft_3d_t2_cufinufft_v2")
def nufft_3d_t2_cufinufft_v2(H_, K_, L_, ugrid, sign, eps, N):
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
    nvtx.RangePushA("extern.nufft_3d_t2_cufinufft_v2:to_gpu")
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
    nvtx.RangePushA("extern.nufft_3d_t2_cufinufft_v2:GPUArray")

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
    nvtx.RangePushA("extern.nufft_3d_t2_cufinufft_v2:setup_cufinufft")
    plan = cufinufft(
            2, shape, 1, eps, isign=-1, dtype=dtype,
            gpu_method=1, gpu_device_id=dev_id)
    plan.set_pts(H_gpu, K_gpu, L_gpu)
    plan.execute(nuvect_gpu, ugrid_gpu)
    nvtx.RangePop()
    #
    #--------------------------------------------------------------------------

    # Copy result back to host -- if the incoming data was on host
    nvtx.RangePushA("extern.nufft_3d_t2_cufinufft_v2:get")
    if not isinstance(H_, GPUArray):
        nuvect = nuvect_gpu.get()
    else:
        nuvect = nuvect_gpu
    nvtx.RangePop()

    return nuvect

