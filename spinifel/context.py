#!/usr/bin/env python
# -*- coding: utf-8 -*-


from atexit         import register
from importlib.util import find_spec
from callmonitor    import intercept as cm_intercept
from functools      import wraps

from .settings import SpinifelSettings
from .utils    import Singleton



class SpinifelContexts(metaclass=Singleton):

    def __init__(self):
        self._rank = 0
        self._comm = None
        self._mpi_initialized = False
        self._dev_id = 0
        self._cuda_initialized = False


    def init_mpi(self):
        if self._mpi_initialized:
            return

        from mpi4py import MPI

        self._comm = MPI.COMM_WORLD
        self._rank = self.comm.Get_rank()

        register(MPI.Finalize)

        settings = SpinifelSettings()
        if settings.verbose:
            print(f"MPI has been initialized on rank {self.rank}")

        self._mpi_initialized = True


    def init_cuda(self):
        if self._cuda_initialized:
            return

        import pycuda
        import pycuda.driver as drv

        drv.init()

        settings     = SpinifelSettings()
        self._dev_id = self.rank%settings.devices_per_node

        dev = drv.Device(self.dev_id)
        ctx = dev.make_context() 

        register(ctx.pop)

        settings = SpinifelSettings()
        if settings.verbose:
            print(f"Rank {self.rank} has been assigned to device {self.dev_id}")

        self._cuda_initialized = True


    @property
    def rank(self):
        return self._rank


    @property
    def comm(self):
        return self._comm


    @property
    def dev_id(self):
        return self._dev_id


    @property
    def cufinufft_available(self):
        loader = find_spec("cufinufft")
        return loader is not None


    @property
    def finufftpy_available(self):
        loader = find_spec("finufftpy")
        return loader is not None



class Profiler(object, metaclass=Singleton):

    def __init__(self):
        self._callmonitor_enabled = False


    @property
    def callmonitor_enabled(self):
        return self._callmonitor_enabled


    @callmonitor_enabled.setter
    def callmonitor_enabled(self, val):
        self._callmonitor_enabled = val


    @property
    def intercept(self):

        def noop(f):
            @wraps(f)
            def _noop(*args, **kwargs):
                return f(*args, **kwargs)

            return _noop

        if self.callmonitor_enabled:
            return cm_intercept
        else:
            return noop
