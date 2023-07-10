#!/bin/bash
#SBATCH -A CSC304
#SBATCH -t 0:29:59
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -J RunSpinifel
#SBATCH -o RunSpinifel_o.%J
#SBATCH -e RunSpinifel_e.%J

set +x

t_start=`date +%s`

# spinifel
source setup/env.sh

export PYTHONPATH=$PYTHONPATH:/ccs/home/petermc/spiral-python-mine
echo "PYTHONPATH"
printenv PYTHONPATH

export test_data_dir="/lustre/orion/chm137/proj-shared/testdata"
export out_dir="/lustre/orion/chm137/scratch/${USER}/${CI_PIPELINE_ID}/spinifel_output"

# Creates the output folder if not already exist.
if [ ! -d "${out_dir}" ]; then
    mkdir -p ${out_dir}
fi

export SPIRAL_HOME=/ccs/home/petermc/spiral-software

export PATH=$PATH:$SPIRAL_HOME/bin
echo "PATH"
printenv PATH

#export SLURM_CPU_BIND="cores"
srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus=1  python -m spinifel --default-settings=frontier_quickstart_petermc.toml --mode=mpi

t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end
