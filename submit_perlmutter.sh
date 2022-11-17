#!/bin/bash
#SBATCH -A m3890_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:20:00
#SBATCH -N 1
##SBATCH --ntasks-per-node=4
##SBATCH -c 32
#SBATCH -J RunSpinifel
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err


export SLURM_CPU_BIND="cores"


t_start=`date +%s`


# Setup spinifel 
source ./setup/env.sh
module load cudatoolkit
export test_data_dir="${CFS}/m2859/data/testdata"
export out_dir="${SCRATCH}/spinifel_output".     # output directory (create one if not exist)
if [ ! -d "${out_dir}" ]; then
    mkdir -p ${out_dir}
fi


set -x


# Setup scaling resources
n_nodes=1
n_bd_cores=1
n_gpus_per_node=4


# Calculate no. tasks for psana2 (we need to reserve two more cores for Smd0 and EventBuilder)
n_tasks=$(($n_bd_cores+2))          


# To make sure that each bd core has one gpu assigned exlusively
n_gpus=$(($n_gpus_per_node*$n_nodes))                           


# Check the allocations
#srun -N${n_nodes} -n${n_tasks} -G${n_gpus} ./gpus_for_tasks


# Run spinifel
#srun -N${n_nodes} -n${n_tasks} -G${n_gpus} python -m spinifel --default-settings=perlmutter_converge.toml --mode=psana2 data.out_dir=$out_dir algorithm.N_images_max=10000 psana.enable=true psana.ps_dir="/global/cfs/cdirs/m2859/data/3iyf/xtc2" algorithm.N_generations=20 runtime.chk_convergence=false algorithm.N_batch_size=1000 psana.ps_parallel=mpi


# Or Run spinifel w/o checkpoint and output only write out mrc
#srun -N${n_nodes} -n${n_tasks} -G${n_gpus} python -m spinifel --default-settings=perlmutter_converge.toml --mode=psana2 data.out_dir=$out_dir algorithm.N_images_max=10000 psana.enable=true psana.ps_dir="/global/cfs/cdirs/m2859/data/3iyf/xtc2" algorithm.N_generations=20 runtime.chk_convergence=false algorithm.N_batch_size=1000 psana.ps_parallel=mpi debug.checkpoint=false debug.verbose=true debug.show_image=false

srun -N${n_nodes} -n${n_tasks} -G${n_gpus} python -m spinifel --default-settings=perlmutter_converge.toml --mode=psana2 data.out_dir=$out_dir algorithm.N_images_max=1000 psana.enable=true psana.ps_dir="/global/cfs/cdirs/m2859/data/3iyf/xtc2" algorithm.N_generations=20 runtime.chk_convergence=false algorithm.N_batch_size=1000 psana.ps_parallel=mpi debug.checkpoint=false debug.verbose=true debug.show_image=false runtime.N_images_per_rank=1000


set +x


t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end

