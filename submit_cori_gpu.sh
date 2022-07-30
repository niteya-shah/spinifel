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
export PYTHONPATH=/global/u1/p/petermc/spiral-python-mine:/opt/mods/lib/python3.8/site-packages
# Hm, I had "-g 0" at end of this srun command. Does it do anything?
srun python -m spinifel --default-settings=cgpu_mpi.toml --mode=mpi

t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end


