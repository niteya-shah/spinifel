# spinifel
Applies the M-TIP algorithm to SPI data.

## Installation
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

## Git clone through SSH
The `build_from_scratch.sh` script clones some directories through ssh.
For this to work properly, you need to have access to the repositories (if they are private) and to have the SSH key corresponding to the computer you are using registered on your GitHub account.

## Data
Spinifel needs input data.
By default, it expects the data to be located in the `$SCRATCH/spinifel_data` directory.
As of writing, the reference file is `2CEX-10k-2.h5` and is available in the `$CFS/m2859/dujardin/spi` directory.
Create the `spinifel_data` directory in your scratch system and copy the data file.

## CUDA Modules
'orientation_matching.cu' file is added to spinifel/sequential/ folder, and changes are made in spinifel/sequential/orientation_matching.py to import CUDA C code from orientation_matching.cu.
These CUDA changes replace the 'euclidean_distances' function from Sklearn, and 'argmin' from numpy packages (in spinifel/sequential/orientation_matching.py file) with handwritten CUDA C kernels (in spinifel/sequential/orientation_matching.cu file) invoked from python through pybind11 package.

## Run on Summit
'run_summit.sh' file is included in scripts/ path. The script file is written for Spinifel MPI version. Spinifel Legion version is not yet tested with CUDA modules. Many of the parameters are hardcoded (for now).

An additional flag '-c' is included in the script. 
CUDA C code for nearest neighbors (Euclidean distance and Heap sort) is executed when the flag is set.
Sklearn library function for nearest neighbors (Euclidean distance and np.argmin) is executed if the flag is not passed as an command argument.

bsub command to run CUDA C code: bsub -P CHM137 -J fxs -W 2:00 -nnodes 1 -e error.log -o output.log "sh scripts/run_summit.sh -m -c -n 1 -t 1 -d 1"

bsub command to run Sklearn code: bsub -P CHM137 -J fxs -W 2:00 -nnodes 1 -e error.log -o output.log "sh scripts/run_summit.sh -m -n 1 -t 1 -d 1"

## Bugs and Issues
Build:
1) The GCC compiler version is changed to 6.4.0. When executing the script by enable CUDA through 'export USE_CUDA=${USE_CUDA:-1}', it results in build errors as it does not support any GCC version greater 7.0.

Execution:
1) Line 15 in spinifel/image.py which contains 'plt.savefig(parms.out_dir / filename)' instruction is commented out while running on Summit, as it is producing a matplotlib package error. The error is being raised only on Summit runs (Code works fine on Cori with the instruction).
