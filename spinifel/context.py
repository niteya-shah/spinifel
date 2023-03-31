#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Manages Global Contexts, eg MPI and CUDA"""


from atexit import register
from importlib.util import find_spec
from functools import wraps
from callmonitor import intercept as cm_intercept

from .settings import SpinifelSettings
from .utils import Singleton, Logger

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


def goodbye():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm.Barrier()
    # MPI.Finalize()


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
        self.ctx = None
        self.settings = SpinifelSettings()
        self.logger = Logger(True, self.settings)

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

        self._psana_excl_ranks = []
        if self.settings.use_psana:
            # A _comm_compute is created here to include only the worker ranks.
            from psana.psexp.tools import (
                get_excl_ranks,
            )  # FIXME: only available on latest psana2

            self._comm = MPI.COMM_WORLD
            self._psana_excl_ranks = get_excl_ranks()

            self._grp_compute = self._comm.group.Excl(self._psana_excl_ranks)
            self._comm_compute = self._comm.Create(self._grp_compute)
            self._rank = self._comm.Get_rank()
        else:
            self._comm = MPI.COMM_WORLD
            self._rank = self._comm.Get_rank()

        # register(MPI.Finalize)
        register(goodbye)
        self.logger.log(f"MPI will be finalized on rank {self.rank}", level=1)
        self._mpi_initialized = True
        self.logger.log(f"MPI has been initialized on rank {self.rank}", level=1)

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

        self._dev_id = self.rank % drv.Device.count()

        dev = drv.Device(self.dev_id)
        self.ctx = dev.retain_primary_context()

        if self.settings.mode != "legion" and self.settings.mode != "legion_psana2":
            self.ctx.push()
            register(self.ctx.pop)

        self.logger.log(
            f"Rank {self.rank} assigned to device {self.dev_id} (total devices: {drv.Device.count()})",
            level=1
        )

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
    def comm_compute(self):
        """
        Get MPI Compute Only Communicator.

        We use `_psana_excl_ranks` to check if psana2 is used
        to avoid intializing SpinfielSettings again.

        For MPI-hdf5, this is world communication.
        For MPI-psana2, this comm exludes all exclusive
        ranks (smd0, eb and srv) from the communication group.
        """
        if self._psana_excl_ranks:
            return self._comm_compute
        else:
            return self._comm

    @property
    def is_worker(self):
        """
        Determines if this MPI rank is a worker rank.

        For MPI-hdf5, all ranks are worker ranks.
        For MPI-psana2, all ranks except the exclusive ranks are
        worker rank.
        """
        if self._rank in self._psana_excl_ranks:
            return False
        else:
            return True

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
