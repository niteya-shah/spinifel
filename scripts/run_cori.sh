#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --qos=premium
#SBATCH --constraint=haswell
#SBATCH --mail-type=ALL
#SBATCH --account=m2859

root_dir="$PWD"

source "$root_dir"/setup/env.sh

export PYTHONPATH="$PYTHONPATH:$root_dir"

export DATA_DIR=$SCRATCH/spinifel_data

export OUT_DIR=$SCRATCH/spinifel_output_job_$SLURM_JOB_ID
mkdir -p $OUT_DIR

export OMP_NUM_THREADS=1
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"

nodes=$SLURM_JOB_NUM_NODES

sockets=2
cores=10
srun -n 1 legion_python main.py -ll:py 1
