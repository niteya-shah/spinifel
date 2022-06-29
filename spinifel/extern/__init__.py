#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Manages external libraries"""



from   importlib.metadata import version

from spinifel       import SpinifelSettings, SpinifelContexts, Profiler
from .util          import CUFINUFFTRequiredButNotFound, \
                           CUFINUFFTVersionUnsupported, \
                           FINUFFTPYRequiredButNotFound, \
                           FINUFFTPYVersionUnsupported

#______________________________________________________________________________
# Load global settings, and contexts
#

settings = SpinifelSettings()
context  = SpinifelContexts()
profiler = Profiler()



#______________________________________________________________________________
# Load cufiNUFFT or fiNUFFTpy depending on settings: use_cuda, use_cufinufft
#

if settings.use_cuda and settings.use_cufinufft:
    # TODO: only manage MPI via contexts! But let's leave this here for now
    if settings.mode == "mpi":
        context.init_mpi()  # Ensures that MPI has been initalized
    context.init_cuda() # this must be called _after_ init_mpi
    if context.cufinufft_available:
        if version("cufinufft") not in ["1.1", "1.2"]:
            raise CUFINUFFTVersionUnsupported
    else:
        raise CUFINUFFTRequiredButNotFound
else:
    if context.finufftpy_available:
        if version("finufft") not in ["1.1.2", "2.1.0"]:
            raise FINUFFTPYVersionUnsupported
    else:
        raise FINUFFTPYRequiredButNotFound

# from . import NUFFT.NUFFT