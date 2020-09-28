# spinifel
Applies the M-TIP algorithm to SPI data.


## Installation
As of writing, spinifel is designed to work on Cori Haswell only.

To install spinifel, clone the repository and run
```
./setup/build_from_scratch.sh
```
from its base directory.

This will create a conda environment, install everything you need inside, and
produce a `env.sh` file to set it up when needed.  To use that environment, use
```
source setup/env.sh
```

### Git clone through SSH
The `build_from_scratch.sh` script clones some directories through ssh.  For
this to work properly, you need to have access to the repositories (if they are
    private) and to have the SSH key corresponding to the computer you are
using registered on your GitHub account.


## Data
Spinifel needs input data.  By default, it expects the data to be located in
the `$SCRATCH/spinifel_data` directory.  As of writing, the reference file is
`2CEX-10k-2.h5` and is available in the `$CFS/m2859/dujardin/spi` directory.
Create the `spinifel_data` directory in your scratch system and copy the data
file.


## CUDA Support

Currently the following components enable CUDA support:
1. `finufft` -> `cufinufft`


### Using `cufinufft`

`cufinufft` uses CUDA to compute the non-uniform FFT. Spinifel will look for
`cufinufft` in the python environment, and -- if it finds it -- it will use
`cufinufft` to compute the `autocorrelation.forward` and
`autocorrelation.adjoint` operations. Since it is not the "default" way to run
Spinifel, it is not automatically installed. To install `cufinufft`, clone:
```
https://github.com/flatironinstitute/cufinufft
```
and follow the installation instructions here: https://github.com/flatironinstitute/cufinufft#advanced-makefile-usage

**Note:** you need to install `cufinufft` to the conda environment used by
Spinifel (by activating it before running `make python`); or alternatively
point the `PYTHONPATH` variable to the `cufinufft` install location. Also: you
will need to add the location of the `cufinufft/lib/libcufinufft.so` to the
`LD_LIBRARY_PATH`.
