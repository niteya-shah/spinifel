#!/bin/bash
#SBATCH -A lcls
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 6:00:00
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

export pdb="5UOE"
export in_fname="${pdb}_rdnOrientation_poissonTrue_increaseFactor100_meshSize152_numberImage10K.h5"
export in_dir="/pscratch/sd/z/zhantao/dgp3d_spi/benchmark/benchmark_dataset"
export network_model_path="/pscratch/sd/z/zhantao/dgp3d_spi/model_resnet_legacy/lightning_logs/version_10450183/checkpoints/epoch=17-step=720.ckpt"


for i in {6..10}
do
    export out_dir="/pscratch/sd/z/zhantao/dgp3d_spi/benchmark/results/${pdb}/useNetworkPrior_false/run_${i}"
    mkdir -p $out_dir
    srun -n1 -G1 python -m spinifel --default-settings=perlmutter_network_benchmark.toml --mode=mpi_network \
        algorithm.use_network_prior=false algorithm.max_intensity_clip=0.005 algorithm.typ_intensity_clip="rel"
    wait
done

for i in {6..10}
do
    export out_dir="/pscratch/sd/z/zhantao/dgp3d_spi/benchmark/results/${pdb}/useNetworkPrior_true/run_${i}"
    mkdir -p $out_dir
    srun -n1 -G1 python -m spinifel --default-settings=perlmutter_network_benchmark.toml --mode=mpi_network \
        algorithm.use_network_prior=true algorithm.max_intensity_clip=0.005 algorithm.typ_intensity_clip="rel"
    wait
done

t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end

