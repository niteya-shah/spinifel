#!/usr/bin/env python
# -*- coding: utf-8 -*-


from . import settings


#_______________________________________________________________________________
# Initialize logging for this module
#

from .utils import getLogger, fully_qualified_module_name
logger = getLogger(fully_qualified_module_name())


#_______________________________________________________________________________
# Run Spinifel
#

if __name__ == "__main__":

    logger.debug(settings)
    logger.info(f"Runtime MODE = {settings.mode}")

    # DON'T use top-level imports here ... sequential, mpi, and legion have
    # incompatible imports for the time being (this might change in future)

    if settings.mode == "sequential":
        from .sequential import main
        main()
    elif settings.mode == "mpi":
        from .mpi import main
        main()
    elif settings.mode == "legion":
        from .legion import main
        main()
    elif settings.mode == "toml":
        print(settings.as_toml())
    else:
        logger.error(f"Didin't do anything, because settings.mode={settings.mode}")