#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Manages external libraries"""



from   importlib.metadata import version
import PyNVTX             as     nvtx

from spinifel       import SpinifelSettings, SpinifelContexts, Profiler
from .util          import CUFINUFFTRequiredButNotFound, \
                           CUFINUFFTVersionUnsupported, \
                           FINUFFTPYRequiredButNotFound, \
                           FINUFFTPYVersionUnsupported
from .util          import transpose


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
    # TODO: only manage MPI via contexts! But let's leave this here for now
    context.init_mpi()  # Ensures that MPI has been initalized
    context.init_cuda() # this must be called _after_ init_mpi
    from pycuda.gpuarray import GPUArray, to_gpu

    if context.cufinufft_available:
        from cufinufft      import cufinufft
        from .cufinufft_ext import nufft_3d_t1_cufinufft_v1, \
                                   nufft_3d_t2_cufinufft_v1, \
                                   nufft_3d_t1_cufinufft_v2, \
                                   nufft_3d_t2_cufinufft_v2
        FINUFFT_CUDA = True
    else:
        raise CUFINUFFTRequiredButNotFound
else:
    if context.finufftpy_available:
        import finufftpy    as     nfft
        from   .finufft_ext import nufft_3d_t1_finufft_v1, \
                                   nufft_3d_t2_finufft_v1

        FINUFFT_CUDA = False
    else:
        raise FINUFFTPYRequiredButNotFound



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
        elif version("cufinufft") >= "1.2":
            nufft_3d_t1 = nufft_3d_t1_cufinufft_v2
            nufft_3d_t2 = nufft_3d_t2_cufinufft_v2
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
