#!/usr/bin/env python
# -*- coding: utf-8 -*-


from . import settings, utils

if __name__ == "__main__":

    logger = utils.Logger(True, settings)
    logger.log(settings, level=1)

    logger.log(f"Runtime MODE = {settings.mode}", level=1)

    # DON'T use top-level imports here ... sequential, mpi, and legion have
    # incompatible imports for the time being (this might change in future)

    if settings.mode == "sequential":
        from .sequential import main

        main()
    elif settings.mode == "mpi":
        # Two input types (hdf5 and xtc2) are supported in mpi mode. 
        # If use_psana is set in toml file, we assume input type is xtc2.
        from .mpi import main

        main()
    elif settings.mode == "mpi_network":
        # Two input types (hdf5 and xtc2) are supported in mpi mode. 
        # If use_psana is set in toml file, we assume input type is xtc2.
        from .mpi_network import main

        main()
    elif settings.mode == "legion":
        from .legion import main

        main()
    elif settings.mode == "toml":
        logger.log(settings.as_toml())
    else:
        logger.log(f"Didin't do anything, because settings.mode={settings.mode}")
