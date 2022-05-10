#!/bin/bash
#SBATCH -A m1759
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -n 8
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
#srun legion_python -ll:py 1 -ll:pyomp 4 -ll:csize 8192 --default-settings=cgpu_mpi.toml --mode=legion

t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end

