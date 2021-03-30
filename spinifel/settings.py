#!/usr/bin/env python
# -*- coding: utf-8 -*-



"""Manage global settings for Spinifel"""



from enum    import Enum, auto
from os      import environ
from inspect import getmembers
from pathlib import Path

from .utils import Singleton



class EnvironType(Enum):
    """
    Enumerates the different kinds of environment variables
    """
    STR  = auto()
    INT  = auto()
    BOOL = auto()
    PATH = auto()



class SpinifelSettings(metaclass=Singleton):
    """
    Singleton Class SpinifelSettings

    Exposes all global settings used by Spinifel
    """

    @staticmethod
    def get_str(x):
        """
        Return string representation of the environment variable `x`
        """
        return environ[x]


    @staticmethod
    def get_bool(x):
        """
        Return boolean representation of the environment variable `x`
        """
        yes = {"1", "on", "true"}
        return environ[x].strip().lower() in yes


    @staticmethod
    def get_int(x):
        """
        Return integer representation of the environment variable `x`
        """
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
        self._devices_per_node = 0
        self._use_cufinufft    = False
        self._ps_smd_n_events  = 0
        self._use_callmonitor  = False

        self._environ = {
                "TEST": ("_test", EnvironType.STR),
                "VERBOSE": ("_verbose", EnvironType.BOOL),
                "DATA_DIR": ("_data_dir", EnvironType.PATH),
                "USE_PSANA": ("_use_psana", EnvironType.BOOL),
                "OUT_DIR": ("_out_dir", EnvironType.PATH),
                "DATA_MULTIPLIER": ("_data_multiplier", EnvironType.INT),
                "VERBOSITY": ("_verbose", EnvironType.INT),
                "SMALL_PROBLEM": ("_small_problem", EnvironType.BOOL),
                "USING_CUDA": ("_using_cuda", EnvironType.BOOL),
                "DEVICES_PER_RS": ("_devices_per_node", EnvironType.INT),
                "USE_CUFINUFFT": ("_use_cufinufft", EnvironType.BOOL),
                "PS_SMD_N_EVENTS": ("_ps_smd_n_events", EnvironType.INT),
                "USE_CALLMONITOR": ("_use_callmonitor", EnvironType.BOOL)
            }

        self.refresh()


    def refresh(self):
        """
        Refresh internal state using environment variables
        """

        for key in self._environ:

            if key not in environ:
                continue

            name, env_type = self._environ[key]

            if env_type == EnvironType.STR:
                env_val = SpinifelSettings.get_str(key)
            elif env_type == EnvironType.INT:
                env_val = SpinifelSettings.get_int(key)
            elif env_type == EnvironType.BOOL:
                env_val = SpinifelSettings.get_bool(key)
            elif env_type == EnvironType.PATH:
                env_val = Path(SpinifelSettings.get_str(key))

            setattr(self, name, env_val)


    def __str__(self):
        propnames = [name for (name, value) in getmembers(self)]
        str_repr  = "SpinifelSettings:\n"
        for prop in propnames:
            if self.isprop(prop):
                str_repr += f"  + {prop}={getattr(self, prop)}\n"
        return str_repr


    def isprop(self, attr):
        """
        Checks if attribute has been decorated using the `@property` decorator
        """
        return isinstance(getattr(type(self), attr, None), property)


    @property
    def test(self):
        """test field used for debugging"""
        return self._test


    @property
    def verbose(self):
        """is verbosity > 0"""
        return self._verbose or self._verbosity > 0


    @property
    def verbosity(self):
        """reporting verbosity"""
        if self._verbosity == 0:
            if self._verbose:
                return 1
            return 0
        return self._verbosity


    @property
    def data_dir(self):
        """data dir"""
        return self._data_dir


    @property
    def use_psana(self):
        """enable PSANA"""
        return self._use_psana


    @property
    def out_dir(self):
        """output dir"""
        return self._out_dir


    @property
    def data_multiplier(self):
        """data multiplier"""
        return self._data_multiplier


    @property
    def small_problem(self):
        """run in small problem mode"""
        return self._small_problem


    @property
    def using_cuda(self):
        """use cuda wherever possible"""
        return self._using_cuda


    @property
    def devices_per_node(self):
        """gpu-device count per node/resource set"""
        return self._devices_per_node


    @property
    def use_cufinufft(self):
        """use cufinufft for nufft support"""
        return self._use_cufinufft


    @property
    def ps_smd_n_events(self):
        """ps smd n events setting"""
        return self._ps_smd_n_events


    @ps_smd_n_events.setter
    def ps_smd_n_events(self, val):
        self._ps_smd_n_events = val
        # update derived environment variable
        environ["PS_SMD_N_EVENTS"] = str(val)


    @property
    def use_callmonitor(self):
        """enable call-monitor"""
        return self._use_callmonitor
