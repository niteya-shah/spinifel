#!/usr/bin/env python
# -*- coding: utf-8 -*-



"""Manage global settings for Spinifel"""



from os      import environ
from inspect import getmembers
from pathlib import Path

from .utils import Singleton




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

        self.__properties = {
            "_test": ("debug", "test", str, "",
                "test field used for debugging"),
            "_verbose": ("debug", "verbose", bool, True,
                "is verbosity > 0"),
            "_verbosity": ("debug", "verbosity", int, 0,
                "reporting verbosity"),
            "_data_dir": ("data", "in_dir", Path, Path(""),
                "data dir"),
            "_data_filename": ("data", "name", str, "",
                "data file name"),
            "_use_psana": ("psana", "enable", bool, False,
                "enable PSANA"),
            "_out_dir": ("data", "out_dir", Path, Path(""),
                "output dir"),
            "_data_multiplier": ("runtime", "data_multiplier", int, 1,
                "data multiplier"),
            "_small_problem": ("runtime", "small_problem", bool, False,
                "run in small problem mode"),
            "_using_cuda": ("runtime", "using_cuda", bool, False,
                "use cuda wherever possible"),
            "_devices_per_node": ("gpu", "devices_per_node", int, 0,
                "gpu-device count per node/resource set"),
            "_use_cufinufft": ("runtime", "use_cufinufft", bool, False,
                "use cufinufft for nufft support"),
            "_ps_smd_n_events": ("psana", "ps_smd_n_events", int, 0,
                "ps smd n events setting"),
            "_use_callmonitor": ("debug", "use_callmonitor", bool, False,
                "enable call-monitor")
        }

        self.__init_internals()

        self._environ = {
            "TEST": ("_test", SpinifelSettings.get_str),
            "VERBOSE": ("_verbose", SpinifelSettings.get_bool),
            "DATA_DIR": ("_data_dir", SpinifelSettings.get_str),
            "USE_PSANA": ("_use_psana", SpinifelSettings.get_bool),
            "OUT_DIR": ("_out_dir", SpinifelSettings.get_str),
            "DATA_MULTIPLIER": ("_data_multiplier", SpinifelSettings.get_int),
            "VERBOSITY": ("_verbose", SpinifelSettings.get_int),
            "SMALL_PROBLEM": ("_small_problem", SpinifelSettings.get_bool),
            "USING_CUDA": ("_using_cuda", SpinifelSettings.get_bool),
            "DEVICES_PER_RS": ("_devices_per_node", SpinifelSettings.get_int),
            "USE_CUFINUFFT": ("_use_cufinufft", SpinifelSettings.get_bool),
            "PS_SMD_N_EVENTS": ("_ps_smd_n_events", SpinifelSettings.get_int),
            "USE_CALLMONITOR": ("_use_callmonitor", SpinifelSettings.get_bool)
        }

        self.refresh()


    def __init_internals(self):
        """
        Set up internal properties based in the _properties spec defined in
        __init__
        """
        for attr in self.__properties:

            _, _, _, default, _ = self.__properties[attr]
            setattr(self, attr, default)







    def refresh(self):
        """
        Refresh internal state using environment variables
        """

        for key in self._environ:

            if key not in environ:
                continue

            name, env_parser = self._environ[key]
            env_val = env_parser(key)

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
        return self._test # noqa: E1101  pylint: disable=no-member


    @property
    def verbose(self):
        """is verbosity > 0"""
        return (self._verbose or      # noqa: E1101 pylint: disable=no-member
                self._verbosity > 0)  # noqa: E1101 pylint: disable=no-member


    @property
    def verbosity(self):
        """reporting verbosity"""
        if self._verbosity == 0:  # noqa: E1101  pylint: disable=no-member
            if self._verbose:     # noqa: E1101 pylint: disable=no-member
                return 1
            return 0
        return self._verbosity # noqa: E1101 pylint: disable=no-member


    @property
    def data_dir(self):
        """data dir"""
        return self._data_dir # noqa: E1101 pylint: disable=no-member


    @property
    def data_filename(self):
        """data file name"""
        return self._data_filename # noqa: E1101 pylint: disable=no-member


    @property
    def use_psana(self):
        """enable PSANA"""
        return self._use_psana # noqa: E1101 pylint: disable=no-member


    @property
    def out_dir(self):
        """output dir"""
        return self._out_dir # noqa: E1101 pylint: disable=no-member


    @property
    def data_multiplier(self):
        """data multiplier"""
        return self._data_multiplier # noqa: E1101 pylint: disable=no-member


    @property
    def small_problem(self):
        """run in small problem mode"""
        return self._small_problem # noqa: E1101 pylint: disable=no-member


    @property
    def using_cuda(self):
        """use cuda wherever possible"""
        return self._using_cuda # noqa: E1101 pylint: disable=no-member


    @property
    def devices_per_node(self):
        """gpu-device count per node/resource set"""
        return self._devices_per_node # noqa: E1101 pylint: disable=no-member


    @property
    def use_cufinufft(self):
        """use cufinufft for nufft support"""
        return self._use_cufinufft # noqa: E1101 pylint: disable=no-member


    @property
    def ps_smd_n_events(self):
        """ps smd n events setting"""
        return self._ps_smd_n_events # noqa: E1101 pylint: disable=no-member


    @ps_smd_n_events.setter
    def ps_smd_n_events(self, val):
        self._ps_smd_n_events = val
        # update derived environment variable
        environ["PS_SMD_N_EVENTS"] = str(val)


    @property
    def use_callmonitor(self):
        """enable call-monitor"""
        return self._use_callmonitor # noqa: E1101 pylint: disable=no-member
