# spinifel
Applies the M-TIP algorithm to SPI data.


## Installation

As of writing, spinifel has been tested on Cori Haswell, Cori GPU, and Summit.

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

This is included in most scripts, so it is not normally necessary to do this
manually.


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
and follow the installation instructions here:
https://github.com/flatironinstitute/cufinufft#advanced-makefile-usage

**Note:** you need to install `cufinufft` to the conda environment used by
Spinifel (by activating it before running `make python`); or alternatively
point the `PYTHONPATH` variable to the `cufinufft` install location. Also: you
will need to add the location of the `cufinufft/lib/libcufinufft.so` to the
`LD_LIBRARY_PATH`.


### Using CUDA Orientation Matching

`orientation_matching.cu` file is added to `spinifel/sequential/` folder, and
changes are made in `spinifel/sequential/orientation_matching.py` to import
CUDA C code from `orientation_matching.cu`.  These CUDA changes replace the
`euclidean_distances` function from `Sklearn`, and `argmin` from `numpy`
packages (in `spinifel/sequential/orientation_matching.py` file) with
handwritten CUDA C kernels (in `spinifel/sequential/orientation_matching.cu`
file) invoked from python through `pybind11` package.

A README file is also provided in the `spinifel/sequential/` path for
`orientation_matching.cu` file.


## Running

### On Summit

`run_summit.sh` file is included in `scripts/` path. The script file is written
for Spinifel MPI version. Spinifel Legion version is not yet tested with CUDA
modules. Many of the parameters are hardcoded (for now).

An additional flag `-c` is included in the script.  CUDA C code for nearest
neighbors (Euclidean distance and Heap sort) is executed when the flag is set.
`Sklearn` library function for nearest neighbors (Euclidean distance and
`np.argmin`) is executed if the flag is not passed as an command argument.

1. `bsub` command to run CUDA C code:
```
bsub -P CHM137 -J fxs -W 2:00 -nnodes 1 -e error.log -o output.log "sh scripts/run_summit.sh -m -c -n 1 -t 1 -d 1"
```

2. `bsub` command to run Sklearn code: 
```
bsub -P CHM137 -J fxs -W 2:00 -nnodes 1 -e error.log -o output.log "sh scripts/run_summit.sh -m -n 1 -t 1 -d 1"
```


## Bugs and Issues


### Build:

1. The GCC compiler version is changed to 6.4.0. When executing the script by
enable CUDA through `export USE_CUDA=${USE_CUDA:-1}`, it results in build
errors as it does not support any GCC version greater 7.0.

2. Installing cufinufft,
cd spinifel/setup
git clone https://github.com/JBlaschke/cufinufft.git
cd cufinufft
echo "CUDA_ROOT=/sw/summit/cuda/10.2.89" >> sites/make.inc.olcf_summit
make site=olcf_summit
export LD_LIBRARY_PATH=${PWD}/lib:${LD_LIBRARY_PATH}
make python

Note: Use cuda 10.2.89 or above then remove pip cache (rm -rf ~/.cache/pip). 



### Execution:

Yay -- none are presently known
