#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Manages Global Contexts, eg MPI and CUDA"""



from atexit         import register
from importlib.util import find_spec
from functools      import wraps
from callmonitor    import intercept as cm_intercept

from .settings import SpinifelSettings
from .utils    import Singleton

MPI4PY_AVAILABLE = False
if find_spec("mpi4py") is not None:
    import mpi4py
    mpi4py.rc(initialize=False, finalize=False)
    from mpi4py import MPI
    MPI4PY_AVAILABLE = True

PYCUDA_AVAILABLE = False
if find_spec("pycuda") is not None:
    # import pycuda
    import pycuda.driver as drv
    PYCUDA_AVAILABLE = True





class SpinifelContexts(metaclass=Singleton):
    """
    Singleton Class SpinifelContexts.

    Singleton: Will be initialized once -- repeted calls to constructor will
    return singleton instance

    Manages global contexts for MPI and CUDA

    Initializers:
        init_mpi
        init_cuda
    """

    def __init__(self):
        self._rank = 0
        self._comm = None
        self._mpi_initialized = False
        self._dev_id = 0
        self._cuda_initialized = False


    def init_mpi(self):
        """
        does nothing if already initialized
        initializes mpi4py and registers MPI.Finalize on atexit stack
        """

        if not MPI4PY_AVAILABLE:
            return

        if self._mpi_initialized:
            return

        # Initialize MPI
        if not MPI.Is_initialized():
            MPI.Init()

        self._comm = MPI.COMM_WORLD
        self._rank = self.comm.Get_rank()

        register(MPI.Finalize)

        settings = SpinifelSettings()
        if settings.verbose:
            print(f"MPI has been initialized on rank {self.rank}")

        self._mpi_initialized = True


    def init_cuda(self):
        """
        does nothing if already initialized
        initializes pycuda and creates context bound to device:
            <MPI Rank> % <Devices Per Resource Set>
        and registers context cleanup on atexit stack
        """

        if not PYCUDA_AVAILABLE:
            return

        if self._cuda_initialized:
            return

        drv.init()

        settings     = SpinifelSettings()
        self._dev_id = self.rank % settings.devices_per_node

        dev = drv.Device(self.dev_id)
        self.ctx = dev.retain_primary_context()

        settings = SpinifelSettings()
        if settings.verbose:
            print(f"Rank {self.rank} assigned to device {self.dev_id}")

        self._cuda_initialized = True


    @property
    def rank(self):
        """
        Get MPI Rank
        """
        return self._rank


    @property
    def comm(self):
        """
        Get MPI Communicator
        """
        return self._comm


    @property
    def dev_id(self):
        """
        Get CUDA device ID
        """
        return self._dev_id


    def cuda_mem_info(self):
        """
        Get CUDA memory info
        Returns gpu_free, gpu_total
        """
        if self._cuda_initialized and PYCUDA_AVAILABLE:
            return drv.mem_get_info()
        return -1, -1


    @property
    def cufinufft_available(self):
        """
        Return true if the cufinufft module is available
        """
        loader = find_spec("cufinufft")
        return loader is not None


    @property
    def finufftpy_available(self):
        """
        Return true if the finufftpy module is available
        """
        loader = find_spec("finufftpy")
        return loader is not None



class Profiler(metaclass=Singleton):
    """
    Singleton Class Profiler

    Singleton: Will be initialized once -- repeted calls to constructor will
    return singleton instance

    Switchable profiling dectorators
    """

    def __init__(self):
        self._callmonitor_enabled = False


    @property
    def callmonitor_enabled(self):
        """
        Controll callmonitor wrapper if set to true
        """
        return self._callmonitor_enabled


    @callmonitor_enabled.setter
    def callmonitor_enabled(self, val):
        self._callmonitor_enabled = val


    @property
    def intercept(self):
        """
        Generate the intercept wrapper: if callmonitor_enabled=False, then this
        wrapper is a no-op
        """

        def noop(func):
            @wraps(func)
            def _noop(*args, **kwargs):
                return func(*args, **kwargs)

            return _noop

        if self.callmonitor_enabled:
            return cm_intercept
        return noop
