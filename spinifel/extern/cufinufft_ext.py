#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Manages external libraries"""


from sys import getsizeof
import numpy as np
import PyNVTX as nvtx
from spinifel import SpinifelSettings, SpinifelContexts, Profiler, Logger
from .util import transpose, CUFINUFFTRequiredButNotFound


# ______________________________________________________________________________
# Load global settings, and contexts
#

settings = SpinifelSettings()
context = SpinifelContexts()
profiler = Profiler()
logger = Logger(True, settings)


# ______________________________________________________________________________
# Load cufiNUFFT or fiNUFFTpy depending on settings: use_cuda, use_cufinufft
#

if settings.use_cuda and settings.use_cufinufft:
    from . import GPUArray, to_gpu

    if context.cufinufft_available:
        from . import cufinufft
    else:
        raise CUFINUFFTRequiredButNotFound


@profiler.intercept
@nvtx.annotate("extern/cufinufft_ext.py", is_prefix=True)
def pts_to_gpu(data, H_, K_, L_, logger):
    """
    Send data to GPU
    """

    if not isinstance(H_, GPUArray):
        H_gpu = to_gpu(H_)
        gpu_free, gpu_total = context.cuda_mem_info()
        logger.log(
                f"H={getsizeof(H_)/1e9:.2f}GB copied "+"\n"+    \
                f"gpu_free={gpu_free/1e9:.2f}GB "+"\n"+         \
                f"gpu_total={gpu_total/1e9:.2f}GB",
                level=1
        )

        K_gpu = to_gpu(K_)
        gpu_free, gpu_total = context.cuda_mem_info()
        logger.log(
                f"K={getsizeof(K_)/1e9:.2f}GB copied "+"\n"+    \
                f"gpu_free={gpu_free/1e9:.2f}GB "+"\n"+         \
                f"gpu_total={gpu_total/1e9:.2f}GB",
                level=1
        )

        L_gpu = to_gpu(L_)
        gpu_free, gpu_total = context.cuda_mem_info()
        logger.log(
                f"K={getsizeof(L_)/1e9:.2f}GB copied "+"\n"+    \
                f"gpu_free={gpu_free/1e9:.2f}GB "+"\n"+         \
                f"gpu_total={gpu_total/1e9:.2f}GB",
                level=1
        )
        data_gpu = to_gpu(data)
    else:
        H_gpu = H_
        K_gpu = K_
        L_gpu = L_
        data_gpu = data

    return data_gpu, H_gpu, K_gpu, L_gpu


@profiler.intercept
@nvtx.annotate("extern/cufinufft_ext.py", is_prefix=True)
def result_to_cpu(data_gpu, H_):
    """
    Send data to CPU
    """

    if not isinstance(H_, GPUArray):
        data = data_gpu.get()
    else:
        data = data_gpu

    return data


@profiler.intercept
@nvtx.annotate("extern/cufinufft_ext.py", is_prefix=True)
def nufft_3d_t1_cufinufft_v1(H_, K_, L_, nuvect, sign, eps, nx, ny, nz):
    """
    Version 1 of cufiNUFFT 3D type 1
    """

    dim = 3
    shape = (nx, ny, nz)
    dev_id = context.dev_id
    complex_dtype = np.complex128
    dtype = np.float64

    print(f"DEBUG in cufinufft_ext nufft_3d_t1_cufinufft_v1 on shape {shape}, dtype {dtype}, complex_dtype {complex_dtype}")
    if settings.verbosity > 0:
        logger.log(f"Using v1 CUDA to solve the NUFFT 3D T1 on device {dev_id}", level=1)

    # Ensure that H_, K_, and L_ have the same shape
    assert H_.shape == K_.shape == L_.shape

    # The rest of Spinifel uses Fortran coordinate order:
    H_, K_, L_ = transpose(H_, K_, L_)

    # Copy input data to Device (if not already there)
    nuvect_gpu, H_gpu, K_gpu, L_gpu = pts_to_gpu(
        nuvect.astype(complex_dtype), H_, K_, L_, logger
    )

    # Allocate space on Device
    nvtx.RangePushA("extern.nufft_3d_t1_cufinufft_v1:GPUArray")
    ugrid_gpu = GPUArray(shape, dtype=complex_dtype, order="F")
    nvtx.RangePop()

    # __________________________________________________________________________
    # Solve the NUFFT
    #
    nvtx.RangePushA("extern.nufft_3d_t1_cufinufft_v1:setup_cufinufft")

    # Change default NUFFT Behaviour
    adjoint_opts = cufinufft.default_opts(nufft_type=1, dim=dim)
    adjoint_opts.gpu_method = 1  # Override with method 1. The default is 2
    adjoint_opts.cuda_device_id = dev_id

    # Run NUFFT
    # TODO: MONA check here performance dependent on data?
    plan = cufinufft(1, shape, sign, eps, dtype=dtype, opts=adjoint_opts)
    plan.set_pts(H_.shape[0], H_gpu, K_gpu, L_gpu)
    plan.execute(nuvect_gpu, ugrid_gpu)
    nvtx.RangePop()
    #
    # --------------------------------------------------------------------------

    # Copy result back to host -- if the incoming data was on host
    ugrid = result_to_cpu(ugrid_gpu, H_)

    return ugrid


@profiler.intercept
@nvtx.annotate("extern/cufinufft_ext.py", is_prefix=True)
def nufft_3d_t2_cufinufft_v1(H_, K_, L_, ugrid, sign, eps, N):
    """
    Version 1 of cufiNUFFT 3D type 2
    """

    dim = 3
    dev_id = context.dev_id
    complex_dtype = np.complex128
    dtype = np.float64

    print(f"DEBUG in cufinufft_ext nufft_3d_t2_cufinufft_v1 on N={N}, dtype {dtype}, complex_dtype {complex_dtype}")
    logger.log(f"Using v1 CUDA to solve the NUFFT 3D T2 on device {dev_id}", level=1)

    gpu_free, gpu_total = context.cuda_mem_info()
    logger.log(
        f"init gpu_free={gpu_free/1e9:.2f}GB "+"\n"+f"gpu_total={gpu_total/1e9:.2f}GB",
        level=1
    )

    # Ensure that H_, K_, and L_ have the same shape
    assert H_.shape == K_.shape == L_.shape

    # The rest of Spinifel uses Fortran coordinate order:
    H_, K_, L_ = transpose(H_, K_, L_)

    # Copy input data to Device (if not already there)
    ugrid_gpu, H_gpu, K_gpu, L_gpu = pts_to_gpu(
        ugrid.astype(complex_dtype), H_, K_, L_, logger
    )

    # Allocate space on Device
    nvtx.RangePushA("extern.nufft_3d_t2_cufinufft_v1:GPUArray")

    nuvect_gpu = GPUArray(shape=(N,), dtype=complex_dtype)
    gpu_free, gpu_total = context.cuda_mem_info()
    logger.log(
            f"nuvect_gpu={getsizeof(nuvect_gpu)/1e9:.2f}GB "+"\n"+  \
            f"allocated gpu_free={gpu_free/1e9:.2f}GB "+"\n"+       \
            f"gpu_total={gpu_total/1e9:.2f}GB",
            level=1
    )
    nvtx.RangePop()

    # __________________________________________________________________________
    # Solve the NUFFT
    #
    nvtx.RangePushA("extern.nufft_3d_t2_cufinufft_v1:setup_cufinufft")

    # Change default NUFFT Behaviour
    forward_opts = cufinufft.default_opts(nufft_type=2, dim=dim)
    forward_opts.gpu_method = 1  # Override with method 1. The default is 2
    forward_opts.cuda_device_id = dev_id

    # Run NUFFT
    plan = cufinufft(2, ugrid.shape, sign, eps, dtype=dtype, opts=forward_opts)
    plan.set_pts(H_.shape[0], H_gpu, K_gpu, L_gpu)
    plan.execute(nuvect_gpu, ugrid_gpu)
    nvtx.RangePop()
    #
    # --------------------------------------------------------------------------

    # Copy result back to host -- if the incoming data was on host
    nuvect = result_to_cpu(nuvect_gpu, H_)

    return nuvect


@profiler.intercept
@nvtx.annotate("extern/cufinufft_ext.py", is_prefix=True)
def nufft_3d_t1_cufinufft_v2(H_, K_, L_, nuvect, sign, eps, nx, ny, nz):
    """
    Version 2 of cufiNUFFT 3D type 1
    """

    shape = (nx, ny, nz)
    dev_id = context.dev_id
    complex_dtype = np.complex128
    dtype = np.float64

    print(f"DEBUG in cufinufft_ext nufft_3d_t1_cufinufft_v2 on shape {shape}, dtype {dtype}, complex_dtype {complex_dtype}")
    logger.log(f"Using v2 CUDA to solve the NUFFT 3D T1 on device {dev_id}", level=1)

    # Ensure that H_, K_, and L_ have the same shape
    assert H_.shape == K_.shape == L_.shape

    # The rest of Spinifel uses Fortran coordinate order:
    H_, K_, L_ = transpose(H_, K_, L_)

    # Copy input data to Device (if not already there)
    nuvect_gpu, H_gpu, K_gpu, L_gpu = pts_to_gpu(
        nuvect.astype(complex_dtype), H_, K_, L_, logger
    )

    # Allocate space on Device
    nvtx.RangePushA("extern.nufft_3d_t1_cufinufft_v2:GPUArray")
    ugrid_gpu = GPUArray(shape, dtype=complex_dtype, order="F")
    nvtx.RangePop()

    # __________________________________________________________________________
    # Solve the NUFFT
    #
    nvtx.RangePushA("extern.nufft_3d_t1_cufinufft_v2:setup_cufinufft")
    # TODO: MONA check here performance dependent on data?
    plan = cufinufft(
        1, shape, 1, eps, isign=sign, dtype=dtype, gpu_method=1, gpu_device_id=dev_id
    )
    plan.set_pts(H_gpu, K_gpu, L_gpu)
    plan.execute(nuvect_gpu, ugrid_gpu)
    nvtx.RangePop()
    #
    # --------------------------------------------------------------------------

    # Copy result back to host -- if the incoming data was on host
    ugrid = result_to_cpu(ugrid_gpu, H_)

    return ugrid


@profiler.intercept
@nvtx.annotate("extern/cufinufft_ext.py", is_prefix=True)
def nufft_3d_t2_cufinufft_v2(H_, K_, L_, ugrid, sign, eps, N):
    """
    Version 2 of cufiNUFFT 3D type 2
    """

    dev_id = context.dev_id
    complex_dtype = np.complex128
    dtype = np.float64

    print(f"DEBUG in cufinufft_ext nufft_3d_t2_cufinufft_v2 on N={N}, dtype {dtype}, complex_dtype {complex_dtype}")
    logger.log(f"Using v2 CUDA to solve the NUFFT 3D T2 on device {dev_id}", level=1)

    gpu_free, gpu_total = context.cuda_mem_info()
    logger.log(
        f"init gpu_free={gpu_free/1e9:.2f}GB "+"\n"+f"gpu_total={gpu_total/1e9:.2f}GB",
        level=1
    )

    # Ensure that H_, K_, and L_ have the same shape
    assert H_.shape == K_.shape == L_.shape

    # The rest of Spinifel uses Fortran coordinate order:
    H_, K_, L_ = transpose(H_, K_, L_)

    # Copy input data to Device (if not already there)
    ugrid_gpu, H_gpu, K_gpu, L_gpu = pts_to_gpu(
        ugrid.astype(complex_dtype), H_, K_, L_, logger
    )

    # Allocate space on Device
    nvtx.RangePushA("extern.nufft_3d_t2_cufinufft_v2:GPUArray")

    nuvect_gpu = GPUArray(shape=(N,), dtype=complex_dtype)
    gpu_free, gpu_total = context.cuda_mem_info()
    logger.log(
            f"nuvect_gpu={getsizeof(nuvect_gpu)/1e9:.2f}GB "+"\n"+  \
            f"allocated gpu_free={gpu_free/1e9:.2f}GB "+"\n"+       \
            f"gpu_total={gpu_total/1e9:.2f}GB",
            level=1
    )
    nvtx.RangePop()

    # __________________________________________________________________________
    # Solve the NUFFT
    #
    nvtx.RangePushA("extern.nufft_3d_t2_cufinufft_v2:setup_cufinufft")
    plan = cufinufft(
        2,
        ugrid.shape,
        1,
        eps,
        isign=sign,
        dtype=dtype,
        gpu_method=1,
        gpu_device_id=dev_id,
    )
    plan.set_pts(H_gpu, K_gpu, L_gpu)
    plan.execute(nuvect_gpu, ugrid_gpu)
    nvtx.RangePop()
    #
    # --------------------------------------------------------------------------

    # Copy result back to host -- if the incoming data was on host
    nuvect = result_to_cpu(nuvect_gpu, H_)

    return nuvect
