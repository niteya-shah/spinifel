#!/usr/bin/env python
# -*- coding: utf-8 -*-


from os      import environ
from inspect import getmembers
from pathlib import Path
from atexit  import register

from .utils import Singleton



class SpinifelSettings(metaclass=Singleton):

    yes = {"1", "on", "true"}


    def get_str(self, x):
        return environ[x]


    def get_bool(self, x):
        return environ[x].strip().lower() in self.yes


    def get_int(self, x):
        return int(environ[x])


    def __init__(self):
        self._test             = ""
        self._verbose          = False
        self._verbosity        = 0
        self._data_dir         = Path("")
        self._use_psana        = False
        self._out_dir          = Path("")
        self._data_multiplier  = 1
        self._small_problem    = False
        self._using_cuda       = False
        self._ranks_per_device = 0

        self.refresh()


    def refresh(self):
        if "TEST" in environ:
            self._test = self.get_str("TEST")

        if "VERBOSE" in environ:
            self._verbose = self.get_bool("VERBOSE")

        if "DATA_DIR" in environ:
            self._data_dir = Path(self.get_str("DATA_DIR")) 

        if "USE_PSANA" in environ:
            self._use_psana = self.get_bool("USE_PSANA")

        if "OUT_DIR" in environ:
            self._out_dir = Path(self.get_str("OUT_DIR"))

        if "DATA_MULTIPLIER" in environ:
            self._data_multiplier = self.get_int("DATA_MULTIPLIER")

        if "VERBOSITY" in environ:
            self._verbosity = self.get_int("VERBOSITY")

        if "SMALL_PROBLEM" in environ:
            self._small_problem = self.get_bool("SMALL_PROBLEM")

        if "USING_CUDA" in environ:
            self._using_cuda = self.get_bool("USING_CUDA")

        if "DEVICES_PER_NODE" in environ:
            self._devices_per_node = self.get_int("DEVICES_PER_NODE")


    def __str__(self):
        propnames = [name for (name, value) in getmembers(self)]
        str_repr  = f"SpinifelSettings:\n"
        for prop in propnames:
            if self.isprop(prop):
                str_repr +=f"  + {prop}={getattr(self, prop)}\n"
        return str_repr


    def isprop(self, attr):
        return isinstance(getattr(type(self), attr, None), property)


    @property
    def test(self):
        return self._test


    @property
    def verbose(self):
        return self._verbose or self._verbosity > 0


    @property
    def verbosity(self):
        if self._verbosity == 0:
            if self._verbose:
                return 1
        else:
            return self._verbosity


    @property
    def data_dir(self):
        return self._data_dir


    @property
    def use_psana(self):
        return self._use_psana


    @property
    def out_dir(self):
        return self._out_dir


    @property
    def data_multiplier(self):
        return self._data_multiplier


    @property
    def small_problem(self):
        return self._small_problem


    @property
    def using_cuda(self):
        return self._using_cuda


    @property
    def devices_per_node(self):
        return self._devices_per_node



class SpinifelContexts(metaclass=Singleton):

    def __init__(self):

        self._rank   = 0
        self._comm   = None
        self._dev_id = 0;


    def init_mpi(self):
        from mpi4py import MPI

        self._comm = MPI.COMM_WORLD
        self._rank = self.comm.Get_rank()

        register(MPI.Finalize)

        settings = SpinifelSettings()
        if settings.verbose:
            print(f"MPI has been initialized on rank {self.rank}")


    def init_cuda(self):
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


    @property
    def rank(self):
        return self._rank


    @property
    def comm(self):
        return self._comm


    @property
    def dev_id(self):
        return self._dev_id