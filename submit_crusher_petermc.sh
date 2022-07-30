#!/bin/bash
#SBATCH -A CSC304
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -J RunSpinifel
#SBATCH -o RunSpinifel.%J
#SBATCH -e RunSpinifel.%J

t_start=`date +%s`

# spinifel
source setup/env.sh

printenv PYTHONPATH

export PYTHONPATH=$PYTHONPATH:/ccs/home/petermc/spiral-python-mine

printenv PYTHONPATH

#export SLURM_CPU_BIND="cores"
srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus=1  python -m spinifel --default-settings=crusher_quickstart_petermc.toml --mode=mpi

t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end
