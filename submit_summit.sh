#!/bin/bash
#BSUB -P CHM137
#BSUB -W 02:00
#BSUB -nnodes 1
#BSUB -alloc_flags gpumps
#BSUB -J RunSpinifel
#BSUB -o RunSpinifel.%J
#BSUB -e RunSpinifel.%J

t_start=`date +%s`

# spinifel
source setup/env.sh
export CUPY_CACHE_DIR=/tmp # used to be $PWD/setup/cupy
export MPLCONFIGDIR=/tmp # writable directory for matplotlib
export PYCUDA_CACHE_DIR=/tmp 

jsrun -n 6 -a 1 -c 7 -g 1 -b rs -d packed python -m spinifel --default-settings=summit_quickstart.toml --mode=mpi

t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end

