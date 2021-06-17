#!/usr/bin/env python
# -*- coding: utf-8 -*-


from .settings import SpinifelSettings
from .context  import SpinifelContexts, Profiler



# Initialize the state of the SpinifelSettings/Contexts singleton classes
settings = SpinifelSettings()
context  = SpinifelContexts()
profiler = Profiler()

# Configure profiler
profiler.callmonitor_enabled = settings.use_callmonitor
