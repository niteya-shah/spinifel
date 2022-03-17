# spinifel

See the Wiki: https://gitlab.osti.gov/mtip/spinifel/-/wikis/home for more informaion.

## CUDA Support

Currently the following components enable CUDA support:
1. Non-Uniform FFT (`finufft`, or `cufinufft`)
2. Orientation Matching

## Continuous Integration

Continuous integration for spinifel on Summit-like machine (Ascent) applies the M-TIP algorithm to SPI data.

CI: https://code.ornl.gov/ecpcitest/chm137/spinifel/-/pipelines (OLCF)

## Interface to FFTX

First download **Spiral** and those of its packages that are required for this application:
* Download **Spiral** at https://github.com/spiral-software/spiral-software  
Check out the `develop` branch, and
set `SPIRAL_HOME` to the directory where you put this.
* Download **spiral-package-fftx** at https://github.com/spiral-software/spiral-package-fftx  
Move it to the directory `$SPIRAL_HOME/namespaces/packages/fftx`, and
check out the `develop` branch.
* Download **spiral-package-simt** at https://github.com/spiral-software/spiral-package-simt  
Move it to the directory `$SPIRAL_HOME/namespaces/packages/simt`, and
check out the `develop` branch.

Now install Spiral following the instructions here:
https://github.com/spiral-software/spiral-software/blob/develop/README.md

Then switch to a new directory and set `SPIRAL_PYTHON_PACKAGES` to this directory.  Download these repositories:
* https://github.com/petermcLBL/python-package-fftx  
Move to the directory `$SPIRAL_PYTHON_PACKAGES/fftx`, and
check out the `develop` branch.
* https://github.com/petermcLBL/python-package-snowwhite  
Move to the directory `$SPIRAL_PYTHON_PACKAGES/snowwhite`, and
check out the `develop` branch.

Set your path to include `$SPIRAL_HOME/bin` so that you can run Spiral.  
And set `PYTHONPATH` to include `$SPIRAL_PYTHON_PACKAGES` so that you can use the Python interfaces to FFTX.

Check out the `development` branch of spinifel, and build it.  
Then on cori, you can run spinifel with:
```
module purge
source setup/env.sh
salloc -N 1 -t 0:30:00 -C gpu --gpus=1 -A m1759 -q special --tasks-per-node=10
srun -n 2 -G 1 python -m spinifel --default-settings=cgpu_quickstart.toml --mode=mpi
```

When the input settings file has `use_fftx = true`, this fork of spinifel runs `fftn` or `ifftn` on *both* FFTX and either NumPy or CuPy as specified by the main repository.  After each call to both routines, spinifel writes out the time taken for each, and the maximum relative difference in the results.  With the setting `use_fftx = false`, FFTX is not called.

With `use_fftx = true`, the first time spinifel executes a call to `fftn` or `ifftn` on an array of a size it has not yet encountered, it will run Spiral to generate C code and compile it to a library that is stored in the directory `$SPIRAL_PYTHON_PACKAGES/snowwhite/.libs`.  Subsequent calls to arrays of the same size will use the library version and not require running Spiral.
