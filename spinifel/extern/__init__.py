#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Manages external libraries"""


from importlib.metadata import version

from spinifel import SpinifelSettings, SpinifelContexts, Profiler, Logger
from .util import (
    CUFINUFFTRequiredButNotFound,
    CUFINUFFTVersionUnsupported,
    FINUFFTPYRequiredButNotFound,
    FINUFFTPYVersionUnsupported,
)

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
if settings.use_cufinufft:
    from cufinufft import cufinufft

if settings.use_cuda and settings.use_cufinufft:
    # TODO: only manage MPI via contexts! But let's leave this here for now
    if settings.mode == "mpi":
        context.init_mpi()  # Ensures that MPI has been initalized
    context.init_cuda() # this must be called _after_ init_mpi
    if settings.use_pygpu:
        from PybindGPU import GPUArray, to_gpu
    else:
        from pycuda.gpuarray import GPUArray, to_gpu

    if context.cufinufft_available:
        from cufinufft import cufinufft
        from .cufinufft_ext import (
            nufft_3d_t1_cufinufft_v1,
            nufft_3d_t2_cufinufft_v1,
            nufft_3d_t1_cufinufft_v2,
            nufft_3d_t2_cufinufft_v2,
        )

        FINUFFT_CUDA = True
    else:
        raise CUFINUFFTRequiredButNotFound
else:
    if context.finufftpy_available:
        import finufftpy as nfft
        from .finufft_ext import nufft_3d_t1_finufft_v1, nufft_3d_t2_finufft_v1

        FINUFFT_CUDA = False
    else:
        raise FINUFFTPYRequiredButNotFound


# ______________________________________________________________________________
# Alias the nufft functions to their cpu/gpu implementations
#

if settings.use_cuda and settings.use_cufinufft:
    logger.log("Orientation Matching: USING_CUDA", level=1)

    if context.cufinufft_available:
        logger.log(f"++++++++++++++++++++: USING_CUFINUFFT version={version('cufinufft')}", level=1)
        if version("cufinufft") == "1.1":
            nufft_3d_t1 = nufft_3d_t1_cufinufft_v1
            nufft_3d_t2 = nufft_3d_t2_cufinufft_v1
        elif version("cufinufft") >= "1.2":
            nufft_3d_t1 = nufft_3d_t1_cufinufft_v2
            nufft_3d_t2 = nufft_3d_t2_cufinufft_v2
        else:
            raise CUFINUFFTVersionUnsupported
    else:
        raise CUFINUFFTRequiredButNotFound
else:
    if context.finufftpy_available:
        logger.log("++++++++++++++++++++: USING_FINUFFTPY", level=1)
        if version("finufftpy") == "1.1.2":
            nufft_3d_t1 = nufft_3d_t1_finufft_v1
            nufft_3d_t2 = nufft_3d_t2_finufft_v1
        else:
            raise FINUFFTPYVersionUnsupported
    else:
        raise FINUFFTPYRequiredButNotFound

if settings.use_cuda and settings.use_cufinufft:
    # TODO: only manage MPI via contexts! But let's leave this here for now
    if settings.mode == "mpi":
        context.init_mpi()  # Ensures that MPI has been initalized
    context.init_cuda()  # this must be called _after_ init_mpi
    if context.cufinufft_available:
        if version("cufinufft") not in ["1.1", "1.2", "1.3"]:
            raise CUFINUFFTVersionUnsupported
    else:
        raise CUFINUFFTRequiredButNotFound
else:
    if context.finufftpy_available:
        if version("finufftpy") not in ["1.1.2", "2.1.0"]:
            raise FINUFFTPYVersionUnsupported
    else:
        raise FINUFFTPYRequiredButNotFound

# from . import NUFFT.NUFFT
