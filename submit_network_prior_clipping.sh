#!/bin/bash
#SBATCH -A lcls
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH -N 1
#SBATCH --gpus-per-task=1
#SBATCH -J RunSpinifel
#SBATCH -o /pscratch/sd/z/zhantao/dgp3d_spi/benchmark/slurm_logs/%x-%j.out
## SBATCH -e /pscratch/sd/z/zhantao/dgp3d_spi/benchmark/slurm_logs/%x-%j.err


t_start=`date +%s`

# spinifel
module load python
source ../spinifel/setup/env.sh
module load cudatoolkit
module load evp-patch

export pdb="6S6Y"
export use_network_prior=true
export in_fname="${pdb}_avoidSymmetricFalse_poissonTrue_increaseFactor100_meshSize152_numberImage5K.h5"
export in_dir="/pscratch/sd/z/zhantao/dgp3d_spi/benchmark/benchmark_dataset"
export network_model_path="/pscratch/sd/z/zhantao/dgp3d_spi/model_resnet_legacy/lightning_logs/version_10073733/checkpoints/epoch=222-step=9366.ckpt"

for i in 0.01 0.02 0.05 0.1 0.2 0.5 1
do
    export out_dir="/pscratch/sd/z/zhantao/dgp3d_spi/benchmark/results/clipping_benchmark/${pdb}/max_${i}"
    mkdir -p $out_dir
    srun -n1 -G1 python -m spinifel --default-settings=perlmutter_clipping_benchmark.toml --mode=mpi_network algorithm.use_network_prior=${use_network_prior} algorithm.max_intensity_clip=$i
    wait
done

t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end

