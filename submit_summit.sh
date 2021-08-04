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
export CUPY_CACHE_DIR=$PWD/setup/cupy

#jsrun -n 6 -a 1 -c 7 -g 1 -b rs -d packed python -m spinifel --default-settings=summit_mpi.toml --mode=mpi
jsrun -n 6 -a 1 -c 7 -g 1 -b rs -d packed legion_python -level lifeline_mapper=1 -ll:py 1 -ll:pyomp 4 -ll:csize 8192 legion_main.py --default-settings=summit_legion.toml --mode=legion

t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end
