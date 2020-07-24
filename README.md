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
