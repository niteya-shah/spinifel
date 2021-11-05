#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Utility for emitting backtraces to help debug freezes.

Usage: modify spinifel/__main__.py or legion_main.py to include the following:

from . import backtrace_helper

or:

import spinifel.backtrace_helper

This should set an automatic timer to dump a backtrace after 3 minutes
(plus some random delay).
"""


import faulthandler
import random


faulthandler.dump_traceback_later(3*60 + 3*random.random())
