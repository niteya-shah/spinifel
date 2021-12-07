#!/bin/bash
#SBATCH -A m1759
#SBATCH -C gpu
#SBATCH -q special
#SBATCH -t 1:00:00
#SBATCH -n 16
#SBATCH --ntasks-per-node=4
#SBATCH -c 10
#SBATCH --gpus-per-task=1
#SBATCH -J RunSpinifel
#SBATCH -o RunSpinifel.%J
#SBATCH -e RunSpinifel.%J

t_start=`date +%s`

# spinifel
source setup/env.sh

export SLURM_CPU_BIND="cores"
srun python -m spinifel --default-settings=cgpu_mpi.toml --mode=mpi

t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end


