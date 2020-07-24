# spinifel
Applies the M-TIP algorithm to SPI data.

## Installation
As of writing, spinifel is designed to work on Cori Haswell only.

To install spinifel, clone the repository and run
```
./setup/build_from_scratch.sh
```
from its base directory.

This will create a conda environment, install everything you need inside, and produce a `env.sh` file to set it up when needed.
To use that environment, use
```
source setup/env.sh
```

## Data
Spinifel needs input data.
By default, it expects the data to be located in the `$SCRATCH/spinifel_data` directory.
As of writing, the reference file is `2CEX-10k-2.h5` and is available in the `$CFS/m2859/dujardin/spi` directory.
Create the `spinifel_data` directory in your scratch system and copy the data file.
