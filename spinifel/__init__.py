#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Spinifel -- SPI FEL analysis tool"""



from .settings import SpinifelSettings
from .context  import SpinifelContexts, Profiler



#______________________________________________________________________________
# Initialize the state of the SpinifelSettings/Contexts singleton classes
#

settings = SpinifelSettings()
context  = SpinifelContexts()
profiler = Profiler()



#______________________________________________________________________________
# Configure profiler
#
profiler.callmonitor_enabled = settings.use_callmonitor
