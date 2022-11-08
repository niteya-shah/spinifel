#!/bin/bash
#SBATCH -A m3890_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -c 32
#SBATCH -N 2
#SBATCH --gres=gpu:4
#SBATCH -J RunSpinifelStrongScaling
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
t_start=`date +%s`
# spinifel
source setup/env.sh
module load cudatoolkit
export PS_PARALLEL='legion'
export PS_SMD_N_EVENTS=8000
nnodes=2
for n in 1 4 8 ; do
    n_images_per_rank=$(( 8000/n ))
    ntasks=$n
    nnodes=2
    ntasks_per_node=4
    if [ $ntasks -lt 4 ]; then
	ntasks_per_node=$n
    fi

    if [ $ntasks -le 4 ]; then
	nnodes=1
    fi
    OUT_DIR_LEGION="${SCRATCH}/spinifel_output_legion/results_strong_${ntasks}tasks"
    mkdir -p $OUT_DIR_LEGION
    export PS_SMD_N_EVENTS=$n_images_per_rank
    echo "Running SPINIFEL strong scaling  with " $ntasks " ranks, n_images_per_rank=" $n_images_per_rank
    srun -N $nnodes --ntasks=$ntasks --ntasks-per-node=$ntasks_per_node -G $ntasks --gpus-per-task=1 --gpu-bind=single:1 legion_python -ll:py 1 -ll:pyomp 4  -ll:csize 32768 legion_main.py --default-settings=perlmutter_converge.toml --mode=legion_psana2 data.out_dir=$OUT_DIR_LEGION algorithm.N_images_max=$n_images_per_rank runtime.N_images_per_rank=$n_images_per_rank psana.enable=true psana.ps_dir="/global/cfs/cdirs/m2859/data/3iyf/xtc2" algorithm.N_generations=20 runtime.chk_convergence=false runtime.cupy_mempool_clear=false algorithm.N_gens_stream=1 algorithm.N_batch_size=1000 debug.verbose=false psana.ps_smd_n_events=$n_images_per_rank psana.batch_size=$n_images_per_rank

    echo "Running SPINIFEL convergence with strong scaling  with " $ntasks " ranks, n_images_per_rank=" $n_images_per_rank
    srun -N $nnodes --ntasks=$ntasks --ntasks-per-node=$ntasks_per_node -G $ntasks --gpus-per-task=1 --gpu-bind=single:1 legion_python -ll:py 1 -ll:pyomp 4  -ll:csize 32768 legion_main.py --default-settings=perlmutter_converge.toml --mode=legion_psana2 data.out_dir=$OUT_DIR_LEGION algorithm.N_images_max=$n_images_per_rank runtime.N_images_per_rank=$n_images_per_rank psana.enable=true psana.ps_dir="/global/cfs/cdirs/m2859/data/3iyf/xtc2" algorithm.N_generations=20 runtime.chk_convergence=true runtime.cupy_mempool_clear=false algorithm.N_gens_stream=1 algorithm.N_batch_size=1000 debug.verbose=false psana.ps_smd_n_events=$n_images_per_rank psana.batch_size=$n_images_per_rank
done
