#!/usr/bin/env python
# -*- coding: utf-8 -*-


from os      import environ
from inspect import getmembers


from .utils import Singleton





class SpinifelSettings(metaclass=Singleton):

    yes = {"1", "on", "true"}

    def __init__(self):
        self._test    = ""
        self._verbose = False

        self.refresh()


    def refresh(self):
        if "TEST" in environ:
            self._test = environ["TEST"]

        if "VERBOSE" in environ:
            self._verbose = environ["VERBOSE"].strip().lower() in self.yes
            


    def __str__(self):
        propnames = [name for (name, value) in getmembers(self)]
        str_repr  = f"SpinifelSettings:\n"
        for prop in propnames:
            if self.isprop(prop):
                str_repr +=f"  + {prop}={getattr(self, prop)}\n"
        return str_repr


    def isprop(self, attr):
        return isinstance(getattr(type(self), attr, None), property)


    @property
    def test(self):
        return self._test


    @property
    def verbose(self):
        return self._verbose
