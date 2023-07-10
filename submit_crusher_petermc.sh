#!/bin/bash
#SBATCH -A CSC304_crusher
#SBATCH -t 0:15:00
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -J RunSpinifel
#SBATCH -o RunSpinifel.%J
#SBATCH -e RunSpinifel.%J

set +x

t_start=`date +%s`

# spinifel
source setup/env.sh

export PYTHONPATH=$PYTHONPATH:/ccs/home/petermc/spiral-python-mine
echo "PYTHONPATH"
printenv PYTHONPATH

export SPIRAL_HOME=/ccs/home/petermc/spiral-software

export PATH=$PATH:$SPIRAL_HOME/bin
echo "PATH"
printenv PATH

#export SLURM_CPU_BIND="cores"
srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus=1  python -m spinifel --default-settings=crusher_quickstart_petermc.toml --mode=mpi

t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end
