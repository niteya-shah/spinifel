#!/usr/bin/env python
# -*- coding: utf-8 -*-



"""Manage global settings for Spinifel"""



from os       import environ
from inspect  import getmembers
from pathlib  import Path
from os.path  import join, abspath, dirname
from toml     import load
from argparse import ArgumentParser

from .utils import Singleton



class MalformedSettingsException(Exception):
    """
    Raise this error whenever settings inputs don't follow the format: `a.b = c`
    """



class CannotProcessSettingsFile(Exception):
    """
    Raise this error whenever the user has not provided a path to a settings
    file, or has provided a settings file that doesn't exists, or has provided
    multiple settings files.
    """



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
            "_verbose": ("debug", "verbose", bool, False,
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

        self.__environ = {
            "PS_SMD_N_EVENTS": ("_ps_smd_n_events", SpinifelSettings.get_int),
        }

        parser = ArgumentParser()
        parser.add_argument("--settings", type=str, nargs=1, default=None)
        parser.add_argument("--default-settings", type=str, nargs=1, default=None)
        parser.add_argument("--mode", type=str, nargs=1, required=True)

        self.__args, self.__params = parser.parse_known_args()

        if (self.__args.settings is None) \
        and (self.__args.default_settings is None):
            raise CannotProcessSettingsFile

        if (self.__args.settings is not None) \
        and (self.__args.default_settings is not None):
            raise CannotProcessSettingsFile

        if self.__args.default_settings is not None:
            self.__toml = join(
                dirname(abspath(__file__)), "..", "settings",
                self.__args.default_settings[0]
            )
        else:
            self.__toml = self.__args.settings[0]

        self.mode = self.__args.mode

        self.refresh()


    def __fget(self, attr):
        """Creates closure for fget lambda"""
        return lambda x: getattr(self, attr)


    def __init_internals(self):
        """
        Set up internal properties based in the _properties spec defined in
        __init__
        """
        for attr in self.__properties:

            _, _, _, default, doc = self.__properties[attr]
            setattr(self, attr, default)

            # Let the user define custon setting function, by defining
            # `@property` members => don't overwrite these with the plain
            # lambda
            if self.isprop(attr[1:]):
                continue

            type.__setattr__(
                type(self), attr[1:],
                property(
                    fget=self.__fget(attr),
                    doc=doc
                )
            )


    def refresh(self):
        """
        Refresh internal state using environment variables
        """

        toml_settings = load(self.__toml)

        for param in self.__params:

            if not ("." in param and "=" in param):
                raise MalformedSettingsException

            setting, val = param.split("=")
            c, k         = setting.split(".")
            toml_settings[c][k] = val

        for attr in self.__properties:

            c, k, parser, _, _ = self.__properties[attr]
            setattr(self, attr, parser(toml_settings[c][k]))

        for key in self.__environ:

            if key not in environ:
                continue

            print(f"WARNING! The environment variable {key} supersedes all "
                  f"other inputs for this setting. If this is unintensional "
                  f"unset {key}.")

            name, env_parser = self.__environ[key]
            env_val = env_parser(key)

            setattr(self, name, env_val)




    def __str__(self):
        propnames = [name for (name, value) in getmembers(self)]
        str_repr  = "SpinifelSettings:\n"
        for prop in propnames:
            if self.isprop(prop):
                c, k, _, _, doc = self.__properties["_" + prop]
                str_repr += f"  + {prop}={getattr(self, prop)}\n"
                str_repr += f"    source: {c}.{k}, description: {doc}\n"
        return str_repr


    def isprop(self, attr):
        """
        Checks if attribute has been decorated using the `@property` decorator
        """
        return isinstance(getattr(type(self), attr, None), property)


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
    def ps_smd_n_events(self):
        """ps smd n events setting"""
        return self._ps_smd_n_events # noqa: E1101 pylint: disable=no-member


    @ps_smd_n_events.setter
    def ps_smd_n_events(self, val):
        self._ps_smd_n_events = val
        # update derived environment variable
        environ["PS_SMD_N_EVENTS"] = str(val)
