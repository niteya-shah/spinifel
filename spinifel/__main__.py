#!/usr/bin/env python
# -*- coding: utf-8 -*-


from . import settings

if __name__ == "__main__":

    if settings.verbose:
        print(settings)

    print(f"Runtime MODE = {settings.mode}")

    # DON'T use top-level imports here ... sequential, mpi, and legion have
    # incompatible imports for the time being (this might change in future)

    if settings.mode == "sequential":
        from .sequential import main
        main()
    elif settings.mode == "mpi":
        from .mpi import main
        main()
    elif settings.mode == "psana2":
        from .mpi import main_psana2
        main_psana2()
    elif settings.mode == "legion":
        from .legion import main
        main()
    elif settings.mode == "toml":
        print(settings.as_toml())
    else:
        print(f"Didin't do anything, because settings.mode={settings.mode}")
