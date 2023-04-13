#!/bin/bash
#SBATCH -A m2859_g
#SBATCH -C gpu
#SBATCH -q early_science
#SBATCH -t 01:00:00
#SBATCH -c 32
#SBATCH -N 4
#SBATCH --gres=gpu:4
#SBATCH -J RunSpinifel
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
t_start=`date +%s`
# spinifel
source setup/env.sh
module load cudatoolkit
n_nodes=4
echo 'Running Legion Psana Streaming with Convergence'
export PS_PARALLEL='legion'
export PS_SMD_N_EVENTS=2500
for n in 1 2 4 ; do
    ntasks=$(( n*4 ))
    echo "Running SPINIFEL with " $ntasks " ranks and N_gens_stream=1"
    OUT_DIR_LEGION="${SCRATCH}/spinifel_output_legion/result_${ntasks}tasks"
    mkdir -p $OUT_DIR_LEGION
    srun -N $n --ntasks=$ntasks --ntasks-per-node=4 -G $ntasks --gpus-per-task=1 --gpu-bind=single:1 legion_python -ll:py 1 -ll:pyomp 1  -ll:csize 32768 legion_main.py --default-settings=perlmutter_converge.toml --mode=legion_psana2 data.out_dir=$OUT_DIR_LEGION algorithm.N_images_max=10000 psana.enable=true psana.ps_dir="/global/cfs/cdirs/m2859/data/3iyf/xtc2" algorithm.N_generations=20 runtime.chk_convergence=false runtime.cupy_mempool_clear=false algorithm.N_gens_stream=1 algorithm.N_batch_size=1000
    echo "Running SPINIFEL with " $ntasks " ranks and N_gens_stream=2"
    srun -N $n --ntasks=$ntasks --ntasks-per-node=4 -G $ntasks --gpus-per-task=1 --gpu-bind=single:1 legion_python -ll:py 1 -ll:pyomp 4  -ll:csize 32768 legion_main.py --default-settings=perlmutter_converge.toml --mode=legion_psana2 data.out_dir=$OUT_DIR_LEGION algorithm.N_images_max=10000 psana.enable=true psana.ps_dir="/global/cfs/cdirs/m2859/data/3iyf/xtc2" algorithm.N_generations=20 runtime.chk_convergence=false runtime.cupy_mempool_clear=false algorithm.N_gens_stream=2 algorithm.N_batch_size=1000
    echo "Running SPINIFEL with " $ntasks " ranks and N_gens_stream=4"
    srun -N $n --ntasks=$ntasks --ntasks-per-node=4 -G $ntasks --gpus-per-task=1 --gpu-bind=single:1 legion_python -ll:py 1 -ll:pyomp 4  -ll:csize 32768 legion_main.py --default-settings=perlmutter_converge.toml --mode=legion_psana2 data.out_dir=$OUT_DIR_LEGION algorithm.N_images_max=10000 psana.enable=true psana.ps_dir="/global/cfs/cdirs/m2859/data/3iyf/xtc2" algorithm.N_generations=20 runtime.chk_convergence=false runtime.cupy_mempool_clear=false algorithm.N_gens_stream=4 algorithm.N_batch_size=1000
    echo "Running SPINIFEL with " $ntasks " ranks and N_gens_stream=2 + convergence=true"
    srun -N $n --ntasks=$ntasks --ntasks-per-node=4 -G $ntasks --gpus-per-task=1 --gpu-bind=single:1 legion_python -ll:py 1 -ll:pyomp 4  -ll:csize 32768 legion_main.py --default-settings=perlmutter_converge.toml --mode=legion_psana2 data.out_dir=$OUT_DIR_LEGION algorithm.N_images_max=10000 psana.enable=true psana.ps_dir="/global/cfs/cdirs/m2859/data/3iyf/xtc2" algorithm.N_generations=20 runtime.chk_convergence=true runtime.cupy_mempool_clear=false algorithm.N_gens_stream=2 algorithm.N_batch_size=1000
t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end
done
