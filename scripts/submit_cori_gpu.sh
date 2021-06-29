#!/bin/bash
#SBATCH -A m1759
#SBATCH -C gpu
#SBATCH -q special
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1

scripts/run_cori.sh -s -m -n
