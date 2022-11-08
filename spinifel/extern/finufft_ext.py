#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Manages external libraries"""


import numpy    as np
import PyNVTX   as nvtx
from   spinifel import SpinifelSettings, SpinifelContexts, Profiler
from   .        import FINUFFTPYRequiredButNotFound



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
    pass
else:
    if context.finufftpy_available:
        from . import nfft
    else:
        raise FINUFFTPYRequiredButNotFound



@profiler.intercept
@nvtx.annotate("extern/finufft_ext.py", is_prefix=True)
def nufft_3d_t1_finufft_v1(x, y, z, nuvect, sign, eps, nx, ny, nz):
    """
    Version 1 of fiNUFFT 3D type 1
    """

    #if settings.verbose:
    #    print("Using CPU to solve the NUFFT 3D T1")

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
@nvtx.annotate("extern/finufft_ext.py", is_prefix=True)
def nufft_3d_t2_finufft_v1(x, y, z, ugrid, sign, eps, n):
    """
    Version 1 of fiNUFFT 3D type 2
    """

    #if settings.verbose:
    #    print("Using CPU to solve the NUFFT 3D T2")

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
