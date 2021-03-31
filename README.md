# spinifel

## Installation

```
git clone -b $BRANCH --recurse-submodules http://gitlab.osti.gov/mtip/spinifel.git
```
Where `$BRANCH` can be development, master or other branch.

As of this writing, spinifel has been tested on Cori Haswell, Cori GPU, and Summit.

To install spinifel, after cloning the repository run:

```
./setup/build_from_scratch.sh
```

from its base directory.

This will create a conda environment, install everything you need inside, and
produce a `env.sh` file to set it up when needed.  To use that environment, use:

```
source setup/env.sh
```

However, the environment setup script `env.sh` is included in most scripts (eg `build_from_scratch.sh`
and the `run_*.sh` scripts), so it is not normally necessary to do this manually unless running spinifel interactively.


## Data

Spinifel needs input data.  By default, it expects the data to be located in
the `$DATA_DIR` directory specified in the `scripts/run_*.sh` scripts.  As of writing,
the reference file is `2CEX-10k-2.h5` and is available on the Summit platform
in `/gpfs/alpine/chm137/proj-shared/data/spi` and on the Cori platform in `$CFS/m2859/dujardin/spi`.

You may create a `spinifel_data` directory in your scratch system and copy the data file.  You may consider putting your output also in this directory.  On Summit, one could create the directory typically as
`/gpfs/alpine/scratch/$USER/chm137/spinifel_data`.  On Cori, it could be created as `$SCRATCH/spinifel_data`.

## CUDA Support

Currently the following components enable CUDA support:
1. Non-Uniform FFT (`finufft`, or `cufinufft`)
2. Orientation Matching


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
The flag `-f` is used to indicate that you want to run cufinufft.

1. `bsub` command to run CUDA C code:
```
bsub -P CHM137 -J spinifel -W 2:00 -nnodes 1 -e error.log -o output.log "sh scripts/run_summit.sh -m -c -n 1 -t 1 -d 1"
```

2. `bsub` command to run Sklearn code: 
```
bsub -P CHM137 -J spinifel -W 2:00 -nnodes 1 -e error.log -o output.log "sh scripts/run_summit.sh -m -n 1 -t 1 -d 1"
```

3. Testing on interactive node with development version*: 
```
./scripts/run_summit_mult.sh -m -n 1 -a 1 -g 1 -r 1 -d 1 -c -f
```
## Continuous Integration

Continuous integration for spinifel on Summit-like machine (Ascent) applies the M-TIP algorithm to SPI data.

CI: https://code.ornl.gov/ecpcitest/chm137/spinifel/-/pipelines (OLCF)


## Developer's Guilde

Here are some helpful tips for developing Spinifel.


### Components for Managing Global Settings and Contexts

These components help manage Spinifel's globa state. We distinguish between
settings and contexts, the latter being used to manage parallelism and devices.


#### `SpinifelSettings`


#### `SpinifelContexts`


## Bugs and Issues


### Build:

1. The GCC compiler version is changed to 6.4.0. When executing the script by
enable CUDA through `export USE_CUDA=${USE_CUDA:-1}`, it results in build
errors as it does not support any GCC version greater 7.0.

2. Installing cufinufft if needing to work with it separately.  Normally this is installed by the build script.
```
cd spinifel/setup
git clone https://github.com/JBlaschke/cufinufft.git
cd cufinufft
echo "CUDA_ROOT=/sw/summit/cuda/10.2.89" >> sites/make.inc.olcf_summit
make site=olcf_summit
export LD_LIBRARY_PATH=${PWD}/lib:${LD_LIBRARY_PATH}
make python
```

Note: Use cuda 10.2.89 or above then remove pip cache (rm -rf ~/.cache/pip). 


### Execution:

Yay -- none are presently known
