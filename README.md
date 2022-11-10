# Spinifel: Single Particle Reconstruction using M-TIP

See the Wiki: https://gitlab.osti.gov/mtip/spinifel/-/wikis/home for more information.

## Trailing Underscore Convention:
Trailing underscores (e.g. rho vs rho_) are used to refer to numpy arrays that have been ifftshifted.

For unshifted arrays, the FFT/IFFT are defined as:  
  * f -> np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(f)))  
  * f -> np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(f)))  
For shifted arrays, the FFT/IFFT are thus as:  
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
