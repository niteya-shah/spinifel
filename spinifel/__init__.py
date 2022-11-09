#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Spinifel -- SPI FEL analysis tool"""


import logging


from .settings import SpinifelSettings

# Frist import of context.py -- this will configure mpi4py.rc to hand off the
# task of initializing and finalizing MPI to the SpinifelContexts class
from .context import SpinifelContexts, Profiler


# ______________________________________________________________________________
# Initialize the state of the SpinifelSettings/Contexts singleton classes
#

settings = SpinifelSettings()
contexts = SpinifelContexts()
profiler = Profiler()


# ______________________________________________________________________________
# Configure profiler
#
profiler.callmonitor_enabled = settings.use_callmonitor


# ______________________________________________________________________________
# Configure Logger
#

logger = logging.getLogger("spinifel.sequential.orientation_matching")
logger = logging.getLogger("spinifel.autocorrelation")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
