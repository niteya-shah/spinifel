#!/bin/bash
#SBATCH -A m1759
#SBATCH -C gpu
#SBATCH -q special
#SBATCH -t 2:00:00
#SBATCH -n 8
#SBATCH --ntasks-per-node=4
#SBATCH -c 20
#SBATCH --gpus-per-task=1
#SBATCH -J RunSpinifel
#SBATCH -o RunSpinifel.%J
#SBATCH -e RunSpinifel.%J

t_start=`date +%s`

# spinifel
source setup/env.sh

export SLURM_CPU_BIND="cores"
#srun python -m spinifel --default-settings=cgpu_legion.toml --mode=legion -g 0
srun legion_python -ll:py 1 -ll:pyomp 8 -ll:csize 16384 legion_main.py --default-settings=cgpu_legion.toml --mode=legion -g 0

t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end


