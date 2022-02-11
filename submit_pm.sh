#!/bin/bash
#SBATCH -A m3890_g
#SBATCH -C gpu
#SBATCH -q early_science
#SBATCH -t 1:00:00
#SBATCH --ntasks-per-node=4
#SBATCH -c 4
#SBATCH -N 4
#SBATCH --gpus-per-task=1
#SBATCH -J RunSpinifel
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err


t_start=`date +%s`

module load cudatoolkit

export IBV_FORK_SAFE=1
export RDMAV_HUGEPAGES_SAFE=1

# spinifel
source setup/env.sh
n_nodes=4
OUT_DIR="${SCRATCH}/spinifel_output/result2_${n_nodes}nodes"
mkdir $OUT_DIR

export SLURM_CPU_BIND="cores"
srun python -m spinifel --default-settings=perlmutter_quickstart.toml --mode=mpi data.out_dir=$OUT_DIR

t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end

