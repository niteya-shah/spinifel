#!/usr/bin/env python
# -*- coding: utf-8 -*-


from . import SpinifelSettings# , Profiler


if __name__ == "__main__":
    settings = SpinifelSettings()
    # profiler = Profiler()

    if settings.verbose:
        print(settings)

    # # Configure profiler
    # profiler.callmonitor_enabled = settings.use_callmonitor
