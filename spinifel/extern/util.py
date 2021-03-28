#!/usr/bin/env python
# -*- coding: utf-8 -*-



"""Common Utilities/Definitions used by the `extern` module"""


import PyNVTX as nvtx



class CUFINUFFTRequiredButNotFound(Exception):
    """Settings require cufiNUFFT, but the module is unavailable"""



class CUFINUFFTVersionUnsupported(Exception):
    """The detected version of cufiNUFFT, is unsupported"""



class FINUFFTPYRequiredButNotFound(Exception):
    """Settings require cufiNUFFT, but the module is unavailable"""



class FINUFFTPYVersionUnsupported(Exception):
    """The detected version of fiNUFFT, is unsupported"""



@nvtx.annotate("extern.transpose")
def transpose(x, y, z):
    """Transposes the order of the (x, y, z) coordinates to (z, y, x)"""
    return z, y, x
