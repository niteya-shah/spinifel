# Spinifel: Single Particle Reconstruction using M-TIP

See the Wiki: https://gitlab.osti.gov/mtip/spinifel/-/wikis/home for more information.

## Trailing Underscore Convention:
Trailing underscores (e.g. rho vs rho_) are used to refer to numpy arrays that have been ifftshifted.

For unshifted arrays (i.e. origin at image center), the FFT/IFFT are defined as:  
  * f -> np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(f)))  
  * f -> np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(f)))  

For shifted arrays (i.e. origin at image corner (0,0)), the FFT/IFFT are thus as:  
  * f_ -> np.fft.fftn(f_)  
  * f_ -> np.fft.ifftn(f_)  

## Black Code Formatting
Developers should run black code formatter before putting in a merge request.  
To install black:  
source setup/env.sh  
pip install black  

To format the code with black:  
black <folder or file>  

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

# Copyright Notice

MTIP Single Particle Imaging (Spinifel) Copyright (c) 2021, The
Regents of the University of California through Lawrence Berkeley
National Laboratory, SLAC National Accelerator Laboratory, and Los
Alamos National Laboratory (subject to receipt of any required
approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.
