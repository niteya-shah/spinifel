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
export test_data_dir="/gpfs/alpine/proj-shared/chm137/data/testdata"
export out_dir=/gpfs/alpine/proj-shared/chm137/${LOGNAME}/spinifel_output
export https_proxy=http://proxy.ccs.ornl.gov:3128/


# Creates the output folder if not already exist.
if [ ! -d "${out_dir}" ]; then
    mkdir -p ${out_dir}
fi


# Run spinifel in mpi-xtc2 (psana2) mode
jsrun -n3 -g1 python -u -m spinifel --default-settings=test_mpi.toml --mode=mpi psana.enable=true


t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end

