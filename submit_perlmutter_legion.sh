#!/bin/bash
#SBATCH -A m2859_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -c 32
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -J RunSpinifel
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
t_start=`date +%s`
# spinifel
source setup/env.sh
module load cudatoolkit
n_nodes=1
echo 'Running Legion Psana2 Streaming/Convergence + Psana2 without streaming/N_conformations=2'
export PS_PARALLEL='legion'
export PS_SMD_N_EVENTS=2500
args="--gpus-per-task=1 --gpu-bind=single:1 legion_python -ll:py 1 -ll:pyomp 1  -ll:csize 32768 legion_main.py --default-settings=perlmutter_converge.toml --mode=legion_psana2 algorithm.N_images_max=10000 psana.enable=true psana.ps_dir="/global/cfs/cdirs/m2859/data/3iyf/xtc2" algorithm.N_generations=4 runtime.cupy_mempool_clear=false algorithm.N_batch_size=1000 debug.verbose=true"
for n in 1 ; do
    ntasks=$(( n*4 ))
    OUT_DIR_LEGION1="${SCRATCH}/spinifel_output_legion/result_${ntasks}tasks_run_1"
    OUT_DIR_LEGION2="${SCRATCH}/spinifel_output_legion/result_${ntasks}tasks_run_2"
    OUT_DIR_LEGION3="${SCRATCH}/spinifel_output_legion/result_${ntasks}tasks_run_3"
    OUT_DIR_LEGION4="${SCRATCH}/spinifel_output_legion/result_${ntasks}tasks_run_4"
    mkdir -p $OUT_DIR_LEGION1
    mkdir -p $OUT_DIR_LEGION2
    mkdir -p $OUT_DIR_LEGION3
    mkdir -p $OUT_DIR_LEGION4
    echo "Running SPINIFEL with " $ntasks " ranks and N_gens_stream=1"
    srun -N $n --ntasks=$ntasks --ntasks-per-node=4 -G $ntasks $args algorithm.N_gens_stream=1 runtime.chk_convergence=false data.out_dir=$OUT_DIR_LEGION1
    echo "Running SPINIFEL with " $ntasks " ranks and N_gens_stream=2"
    srun -N $n --ntasks=$ntasks --ntasks-per-node=4 -G $ntasks $args algorithm.N_gens_stream=2 runtime.chk_convergence=false data.out_dir=$OUT_DIR_LEGION2
    echo "Running SPINIFEL with " $ntasks " ranks and N_gens_stream=2 + convergence=true"
    srun -N $n --ntasks=$ntasks --ntasks-per-node=4 -G $ntasks $args algorithm.N_gens_stream=2 runtime.chk_convergence=true  data.out_dir=$OUT_DIR_LEGION3
    echo "Running SPINIFEL with " $ntasks " ranks and N_gens_stream=1 and N_conformations=2"
    srun -N $n --ntasks=$ntasks --ntasks-per-node=4 -G $ntasks $args algorithm.N_conformations=2 --mode=legion runtime.chk_convergence=false  data.out_dir=$OUT_DIR_LEGION4
    t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end
done
