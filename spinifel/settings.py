#!/usr/bin/env python
# -*- coding: utf-8 -*-



"""Manage global settings for Spinifel"""



from os       import environ
from inspect  import getmembers
from pathlib  import Path
from os.path  import join, abspath, dirname, expandvars
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



class NotAVector(Exception):
    """
    Raise this error whenever trying to parse a vector representation that is
    malformed in any way.
    """



def get_str(x):
    """
    Return string representation of the environment variable `x`
    """
    return environ[x]



def get_path(x):
    """
    Return string representation of the environment variable `x`
    """
    return Path(environ[x])



def str2bool(x):
    """
    Parse a string representation of a boolean
    """
    yes = {"1", "on", "true"}
    return x.lower() in yes



def get_bool(x):
    """
    Return boolean representation of the environment variable `x`
    """
    return str2bool(environ[x].strip())



def get_int(x):
    """
    Return integer representation of the environment variable `x`
    """
    return int(environ[x])



def parse_bool(x):
    """
    Parse string, integer, or boolean representation of a boolean
    """

    # Toml already parses some inputs, so the input might already have the
    # correct format
    if isinstance(x, bool):
        return x


    if isinstance(x, str):
        return str2bool(x)

    return bool(x)



def parse_strvec(x, elt_parser):
    """
    Parse a string representation of a vector. All elements use the same parser
    `elt_parser`. The element parser is only invoked if `x` is a well-formed
    string representation of a vecotr. Vector elements are delimited by commas.
    """

    # Toml already parses some inputs, so the input might already have the
    # correct format
    if isinstance(x, list):
        return tuple(x)

    if isinstance(x, tuple):
        return x

    if not isinstance(x, str):
        raise NotAVector

    if x[0] != "[":
        raise NotAVector

    if x[-1] != "]":
        raise NotAVector

    if x.count("[") != 1:
        raise NotAVector

    if x.count("]") != 1:
        raise NotAVector

    return tuple(
        [elt_parser(y.strip()) for y in x[1:-1].split(",") if len(y) > 0]
    )



def parse_strvec_int(x):
    """
    Narrowing of parse_strvec:
    parse_strvec_int = parse_strvec(___, int)
    """
    return parse_strvec(x, int)



def parse_strvec_float(x):
    """
    Narrowing of parse_strvec:
    parse_strvec_int = parse_strvec(___, float)
    """
    return parse_strvec(x, float)



def parse_strvec_bool(x):
    """
    Narrowing of parse_strvec:
    parse_strvec_int = parse_strvec(___, parse_bool)
    """
    return parse_strvec(x, parse_bool)



class SpinifelSettings(metaclass=Singleton):
    """
    Singleton Class SpinifelSettings

    Exposes all global settings used by Spinifel
    """


    def __init__(self):

        self.__properties = {
            "_test": (
                "debug", "test",
                str, "",
                "test field used for debugging"
            ),
            "_verbose": (
                "debug", "verbose",
                parse_bool, False,
                "is verbosity > 0"
            ),
            "_verbosity": (
                "debug", "verbosity",
                int, 0,
                "reporting verbosity"
            ),
            "_checkpoint": (
                "debug", "checkpoint",
                parse_bool, True,
                "save intermediate checkpoint"
            ),
            "_data_dir": (
                "data", "in_dir",
                Path, Path(""),
                "data dir"
            ),
            "_data_filename": (
                "data", "name",
                str, "",
                "data file name"
            ),
            "_use_psana": (
                "psana", "enable",
                parse_bool, False,
                "enable PSANA"
            ),
            "_out_dir": (
                "data", "out_dir",
                Path, Path(""),
                "output dir"
            ),
            "_N_images_per_rank": (
                "runtime", "N_images_per_rank",
                int, 10,
                "no. of images per rank"
            ),
            "_use_cuda": (
                "runtime", "use_cuda",
                parse_bool, False,
                "use cuda wherever possible"
            ),
            "_devices_per_node": (
                "gpu", "devices_per_node",
                int, 0,
                "gpu-device count per node/resource set"
            ),
            "_use_cufinufft": (
                "runtime", "use_cufinufft",
                parse_bool, False,
                "use cufinufft for nufft support"
            ),
            "_use_cupy": (
                "runtime", "use_cupy",
                parse_bool, False,
                "use cupy wherever possible"
            ),
            "_ps_smd_n_events": (
                "psana", "ps_smd_n_events",
                int, 100,
                "no. of events to be sent to an EventBuilder core"
            ),
            "_ps_eb_nodes": (
                "psana", "ps_eb_nodes",
                int, 1,
                "no. of eventbuilder cores"
            ),
            "_ps_srv_nodes": (
                "psana", "ps_srv_nodes",
                int, 0,
                "no. of server cores"
            ),
            "_ps_exp": (
                "psana", "exp",
                str, "xpptut1",
                "PSANA experiment name"
            ),
            "_ps_batch_size": (
                "psana", "ps_batch_size",
                int, 100,
                "PSANA batch size"
            ),
            "_ps_dir": (
                "psana", "ps_dir",
                str, "",
                "PSANA xtc2 directory"
            ),
            "_ps_parallel": (
                "psana", "ps_parallel",
                str, "legion",
                "Use legion or mpi mode for PSANA"
            ),
            "_ps_runnum": (
                "psana", "runnum",
                int, 1,
                "PSANA experiment number"
            ),
            "_ps_batch_size": (
                "psana", "ps_batch_size",
                int, 100,
                "PSANA batch size"
            ),
            "_ps_dir": (
                "psana", "ps_dir",
                str, "",
                "PSANA xtc2 directory"
            ),
            "_use_callmonitor": (
                "debug", "use_callmonitor",
                parse_bool, False,
                "enable call-monitor"
            ),
            "_use_single_prec": (
                "runtime", "use_single_prec",
                parse_bool, False,
                "if true, spinifel will use single-precision floating point"
            ),
            "_chk_convergence": (
                "runtime", "chk_convergence",
                parse_bool, True,
                "if false, no check if output density converges"
            ),
            "_det_shape": (
                "detector", "shape",
                parse_strvec_int, (4, 512, 512),
                "detector shape"
            ),
            "_N_images_max": (
                "algorithm", "N_images_max",
                int, 10000,
                "max images"
            ),
            "_N_generations": (
                "algorithm", "N_generations",
                int, 10,
                "max generations"
            ),
            "_data_field_name": (
                "detector", "data_field_name",
                str, "intensities",
                "name of data field in the detector output files"
            ),
            "_data_type_str": (
               "detector", "data_type_str",
                str, "float32",
                "type string (numpy) for the detector output"
            ),
            "_pixel_position_shape_0": (
                "algorithm", "pixel_position_shape_0",
                parse_strvec_int, (3,),
                "pixel_position_shape = pixel_position_shape_0 + det_shape"
            ),
            "_pixel_position_type_str": (
                "algorithm", "pixel_position_type_str",
                str, "float32",
                "type string (numpy) for the pixel_position array"
            ),
            "_pixel_index_shape_0": (
                "algorithm", "pixel_index_shape_0",
                parse_strvec_int, (2,),
                "pixel_index_shape = pixel_index_shape_0 + det_shape"
            ),
            "_pixel_index_type_str": (
                "algorithm", "pixel_index_type_str",
                str, "int32",
                "type string (numpy) for the pixel_index array"
            ),
            "_orientation_type_str": (
                "algorithm", "orientation_type_str",
                str, "float32",
                "type string (numpy) for the orientation array"
            ),
            "_volume_type_str": (
                "algorithm", "volume_type_str",
                str, "complex64",
                "type string (numpy) for the volume array"
            ),
            "_volume_shape": (
                "algorithm", "volume_shape",
                parse_strvec_int, (151, 151, 151),
                "shape of volume array"
            ),
            "_oversampling": (
                "algorithm", "oversampling",
                int, 1,
                "oversampling rate"
            ),
            "_solve_ac_maxiter": (
                "algorithm", "solve_ac_maxiter",
                int, 100,
                "max number of iterations in the CG solver"
            ),
            "_beta": (
                "algorithm", "beta",
                float, 0.9,
                "negative feedback in HIO"
            ),
            "_cutoff": (
                "algorithm", "cutoff",
                float, 5e-2,
                "cutoff in shrinkwrap"
            ),
            "_nER": (
                "algorithm", "nER",
                int, 10,
                "number of iterations in ER"
            ),
            "_nHIO": (
                "algorithm", "nHIO",
                int, 5,
                "number of iterations in HIO"
            ),
            "_N_phase_loops": (
                "algorithm", "N_phase_loops",
                int, 5,
                "number of loops for phasing"
            ),
            "_N_clipping": (
                "algorithm", "N_clipping",
                int, 1,
                "N_clipping parameter for dataset preprocessing"
            ),
            "_N_binning": (
                "algorithm", "N_binning",
                int, 4,
                "N_binning parameter for dataset preprocessing"
            ),
            "_N_orientations": (
                "algorithm", "N_orientations",
                int, 1000,
                "N_orientations parameter for orientation matching"
            ),
            "_N_batch_size": (
                "algorithm", "N_batch_size",
                int, 1000,
                "N_batch_size parameter for slicing in batches"
            ),
            "_load_generation": (
                "algorithm", "load_generation",
                int, -1,
                "start from output of this generation"
            ),
            "_fluctuation_analysis": (
                "algorithm", "fluctuation_analysis",
                parse_bool, False,
                "Perform Fluctuation analysis"
            ),
            "_N_image_batches_max": (
                "algorithm", "N_image_batches_max",
                int, 1,
                "Maximum number of image batches to load per iteration"
            ),
            "_must_converge": (
                "runtime", "must_converge",
                parse_bool, False,
                "Algorithm is expected to converge"
            ),
            "_cupy_mempool_clear": (
                "runtime", "cupy_mempool_clear",
                parse_bool, True,
                "Aggresively clear CuPy mempool"
            ),
            "_pdb_path": (
                "fsc", "pdb_path",
                Path, Path(""),
                "Path for the PDB File"
            ),
            "_fsc_zoom": (
                "fsc", "fsc_zoom",
                float, 1.0,
                "Zoom factor during alignment"
            ),
            "_fsc_sigma": (
                "fsc", "fsc_sigma",
                float, 0,
                "Sigma for Gaussian filtering during alignment"
            ),
            "_fsc_niter": (
                "fsc", "fsc_niter",
                int, 10,
                "Number of alignment iterations to run"
            ),
            "_fsc_nsearch": (
                "fsc", "fsc_nsearch",
                int, 360,
                "Number of quaternions to score per iteration"
            ),
            "_fsc_min_cc": (
                "fsc", "fsc_min_cc",
                float, 0.8,
                "Minimum correlation used in convergence test"
            ),
            "_fsc_min_change_cc": (
                "fsc", "fsc_min_change_cc",
                float, 0.001,
                "Minimum change in correlation used in convergence test"
            ),
            "_fsc_fraction_known_orientations": (
                "fsc", "fsc_fraction_known_orientations",
                float, 0.75,
                "Amount of correct orientations in main and unit tests"
            ),
        }

        self.__init_internals()

        self.__environ = {
            "TEST": ("_test", get_str),
            "VERBOSE": ("_verbose", get_bool),
            "DATA_DIR": ("_data_dir", get_path),
            "PDB_PATH": ("_pdb_path", get_path),
            "DATA_FILENAME": ("_data_filename", get_str),
            "USE_PSANA": ("_use_psana", get_bool),
            "OUT_DIR": ("_out_dir", get_path),
            "N_IMAGES_PER_RANK": ("_N_images_per_rank", get_int),
            "VERBOSITY": ("_verbose", get_int),
            "USE_CUDA": ("_use_cuda", get_bool),
            "DEVICES_PER_RS": ("_devices_per_node", get_int),
            "USE_CUFINUFFT": ("_use_cufinufft", get_bool),
            "USE_CUPY": ("_use_cupy", get_bool),
            "PS_SMD_N_EVENTS": ("_ps_smd_n_events", get_int),
            "PS_EB_NODES": ("_ps_eb_nodes", get_int),
            "PS_SRV_NODES": ("_ps_srv_nodes", get_int),
            "PS_PARALLEL": ("_ps_parallel", get_str),
            "USE_CALLMONITOR": ("_use_callmonitor", get_bool),
            "CHK_CONVERGENCE": ("_chk_convergence", get_bool)
        }

        p = ArgumentParser()
        p.add_argument("--settings", type=str, nargs=1, default=None)
        p.add_argument("--default-settings", type=str, nargs=1, default=None)
        p.add_argument("--mode", type=str, nargs=1, required=True)
        p.add_argument("-t","--tag-generation", type=str, default=None)
        p.add_argument("-g","--load-generation", type=int, default=-1)

        self.__args, self.__params = p.parse_known_args()

        self.mode = self.__args.mode[0]
        self.tag_gen = self.__args.tag_generation
        self.load_gen = self.__args.load_generation

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
            if c not in toml_settings:
                toml_settings[c] = {}
            toml_settings[c][k] = val

        for attr in self.__properties:

            c, k, parser, default_val, _ = self.__properties[attr]

            # if property is not found in toml settings, use default
            if c not in toml_settings:
                val = default_val
            else:
                if k not in toml_settings[c]:
                    val = default_val
                else:
                    val = toml_settings[c][k]

            if parser == str or parser == Path:
                val = expandvars(val)

            setattr(self, attr, parser(val))

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
        propnames = (name for (name, value) in getmembers(self))
        str_repr  = "SpinifelSettings:\n"
        for prop in propnames:
            if self.isprop(prop):
                str_repr += f"  + {prop} = {getattr(self, prop)}\n"

                if "_" + prop in self.__properties.keys():
                    c, k, _, _, doc = self.__properties["_" + prop]
                    str_repr += f"    source: {c}.{k}\n"
                    str_repr += f"    description: {doc}\n"
                else:
                    str_repr += f"    (derived data)\n"

        return str_repr


    def as_toml(self):
        propnames = (x[1:] for x in self.__properties.keys())
        str_repr  = "SpinifelSettings Toml Spec:\n"
        categories = dict()
        for prop in propnames:
            # if self.isprop(prop):
            c, k, _, _, _ = self.__properties["_" + prop]
            if c not in categories:
                categories[c] = dict()
            categories[c][k] = prop

        for c in categories:
            str_repr += f"\n[{c}]\n"
            for k in categories[c]:
                prop = categories[c][k]
                _, _, p, _, doc = self.__properties["_" + prop]
                val = getattr(self, prop)

                # reformat special parsed data types
                if (p == str) or (p == Path):
                    val = "\"" + str(val) + "\""
                if p == parse_bool:
                    val = str(val).lower()
                if p in (parse_strvec_int, parse_strvec_float,
                         parse_strvec_bool):
                    val = "[" + str(val).lower()[1:-1] + "]"

                str_repr += f"{k} = {val} "
                str_repr += f" # {doc} ({categories[c][k]})\n"
        return str_repr


    def isprop(self, attr):
        """
        Checks if attribute has been decorated using the `@property` decorator
        """
        return isinstance(getattr(type(self), attr, None), property)


    @property
    def verbose(self):
        """
        is verbosity > 0
        """
        return (self._verbose or      # noqa: E1101 pylint: disable=no-member
                self._verbosity > 0)  # noqa: E1101 pylint: disable=no-member


    @property
    def verbosity(self):
        """
        reporting verbosity
        """
        if self._verbosity == 0:  # noqa: E1101  pylint: disable=no-member
            if self._verbose:     # noqa: E1101 pylint: disable=no-member
                return 1
            return 0
        return self._verbosity # noqa: E1101 pylint: disable=no-member


    @property
    def data_path(self):
        """
        path of data file:
        data_path = self.data_dir / self.data_filename
        """
        return self._data_dir / self._data_filename


    @property
    def ps_smd_n_events(self):
        """
        no. of events to be sent to an EventBuilder core
        """
        return self._ps_smd_n_events # noqa: E1101 pylint: disable=no-member


    @ps_smd_n_events.setter
    def ps_smd_n_events(self, val):
        self._ps_smd_n_events = val
        # update derived environment variable
        environ["PS_SMD_N_EVENTS"] = str(val)


    @property
    def ps_eb_nodes(self):
        """
        no. of event builder cores
        """
        return self._ps_eb_nodes # noqa: E1101 pylint: disable=no-member


    @ps_eb_nodes.setter
    def ps_eb_nodes(self, val):
        self._ps_eb_nodes = val
        # update derived environment variable
        environ["PS_EB_NODES"] = str(val)


    @property
    def ps_srv_nodes(self):
        """
        no. of server cores
        """
        return self._ps_srv_nodes # noqa: E1101 pylint: disable=no-member


    @ps_srv_nodes.setter
    def ps_srv_nodes(self, val):
        self._ps_srv_nodes = val
        # update derived environment variable
        environ["PS_SRV_NODES"] = str(val)

    @property
    def ps_parallel(self):
        """
        ps parallel mode
        """
        return self._ps_parallel

    @ps_parallel.setter
    def ps_parallel(self, val):
        """
        update derived environment variable
        """
        self._ps_parallel = val
        environ["PS_PARALLEL"] = val

    @property
    def pixel_position_shape(self):
        """
        pixel_position_shape_0 + det_shape
        """
        return self._pixel_position_shape_0 + self._det_shape


    @property
    def pixel_index_shape(self):
        """
        pixel_index_shape_0 + det_shape
        """
        return self._pixel_index_shape_0 + self._det_shape


    @property
    def Mquat(self):
        """
        int(self._oversampling * 20)  # 1/4 of uniform grid size
        """
        return int(self._oversampling * 20)


    @property
    def M(self):
        """
        4*Mquat + 1
        """
        return 4 * self.Mquat + 1


    @property
    def M_ups(self):
        """
        Upsampled grid for AC convolution technique
        """
        return 2 * self.M


    @property
    def N_binning_tot(self):
        return self.N_clipping + self.N_binning


    @property
    def reduced_det_shape(self):
        return self.det_shape[:-2] + (
            self.det_shape[-2] // 2**self.N_binning_tot,
            self.det_shape[-1] // 2**self.N_binning_tot
        )


    @property
    def reduced_pixel_position_shape(self):
        return  self.pixel_position_shape_0 + self.reduced_det_shape


    @property
    def reduced_pixel_index_shape(self):
        return self.pixel_index_shape_0 + self.reduced_det_shape
