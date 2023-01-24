# spinifel

See the Wiki: https://gitlab.osti.gov/mtip/spinifel/-/wikis/home for more informaion.


## Effort to port to Crusher (Jan 23, 2023) by Darren Hsu

In this branch we are trying to make sure that the spinifel code runs on Crusher and ultimately gives correct results.

The two main tasks are to (1) wrap CUDA code with a HIP wrapper and (2) replace the PyCUDA dependency by using PybindGPU written by Johannes Blaschke.

For (1), while working on `cufinufft`, we have encountered compilation errors. For now, we use the LLVM compiler provided by Balint Joo, which can be found at `/gpfs/alpine/world-shared/stf006/bjoo/llvm-amd-stg-open-f1937ea` and loaded by
```bash
module use /gpfs/alpine/world-shared/stf006/bjoo/llvm-amd-stg-open-f1937ea
module load amd-llvm
```
This, however, limits us to ROCm/5.1.0. We will have to contact Balint for future ROCm versions or figure out a way to automate this.

For (2) we have worked on `cufinufft` and now have ![a branch that does not depend on PyCUDA](https://github.com/darrenjhsu/cufinufft/tree/djh/PyBindGPU). However, the `spinifel` code itself now uses PyCUDA extensively, and work is being done to replace those calls.

### General strategy to replace PyCUDA calls with PybindGPU and work in progress

Most of these changes are in commit `d744ed0c` of the `djh/crusher_122022`. 

Consulting Johannes's code and examples, I am replacing all pycuda calls like this:

```python
#from pycuda.gpuarray import GPUArray, to_gpu
import PybindGPU
import PybindGPU.gpuarray as gpuarray
from PybindGPU.gpuarray import GPUArray, to_gpu 
```

In `spinifel/extern/nufft_ext.py`, there is an unfinished implmentation for the `PagelockedAllocator` at line 154, and the return `GPUArray` of `class NUFFT` (line 250) does not have an allocator associated with it.



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
