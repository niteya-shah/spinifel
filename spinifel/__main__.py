#!/usr/bin/env python
# -*- coding: utf-8 -*-


from . import SpinifelSettings



if __name__ == "__main__":
    settings = SpinifelSettings()
    if settings.verbose:
        print(settings)
