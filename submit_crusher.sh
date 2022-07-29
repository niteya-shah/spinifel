#!/bin/bash
#SBATCH -A chm137
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -J RunSpinifel
#SBATCH -o RunSpinifel.%J
#SBATCH -e RunSpinifel.%J

t_start=`date +%s`

# spinifel
source setup/env.sh

#export SLURM_CPU_BIND="cores"
srun -N1 -n8 -c8 --gpus=4 python -m spinifel --default-settings=crusher_quickstart_small.toml --mode=mpi
##srun python -m spinifel --default-settings=cgpu_legion.toml --mode=legion -g 0
#srun legion_python -ll:py 1 -ll:pyomp 8 -ll:csize 16384 legion_main.py --default-settings=cgpu_legion.toml --mode=legion -g 0

t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end


