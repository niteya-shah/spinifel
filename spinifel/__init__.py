#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Spinifel -- SPI FEL analysis tool"""


from .settings import SpinifelSettings
# Frist import of context.py -- this will configure mpi4py.rc to hand off the
# task of initializing and finalizing MPI to the SpinifelContexts class
from .context  import SpinifelContexts, Profiler


#_______________________________________________________________________________
# Initialize the state of the SpinifelSettings/Contexts singleton classes
#

settings = SpinifelSettings()
contexts = SpinifelContexts()
profiler = Profiler()


#_______________________________________________________________________________
# Configure profiler
#
profiler.callmonitor_enabled = settings.use_callmonitor