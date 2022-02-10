#!/bin/bash

while getopts "N:n:c:g:f:i:m:o:b:s:" option
do
case "${option}"
in
N) NNODES=$OPTARG;;
n) NTASKS_PER_NODE=$OPTARG;;
c) NCPUS_PER_TASK=$OPTARG;;
g) NGPUS=$OPTARG;;
f) NGPUS_PER_TASK=$OPTARG;;
i) N_IMAGES_PER_RANK=$OPTARG;;
m) RUN_MODE=$OPTARG;;
o) NORIENT=$OPTARG;;
b) NBINNING=$OPTARG;;
s) BATCHSIZE=$OPTARG;;
esac
done

echo "BATCHSIZE: $BATCHSIZE"

root_dir=${root_dir:-"$PWD"}
echo "root_dir: $root_dir"

source "$root_dir"/setup/env.sh

if [[ -z $NNODES ]]; then
        NNODES="1"
fi
echo "NNODES: $NNODES"

if [[ -z $NTASKS_PER_NODE ]]; then
        NTASKS_PER_NODE="1"
fi
echo "NTASKS_PER_NODE: $NTASKS_PER_NODE"

if [[ -z $NCPUS_PER_NODE ]]; then
        NCPUS_PER_NODE="4"
fi
echo "NCPUS_PER_NODE: $NCPUS_PER_NODE"

if [[ -n $RUN_MODE ]]; then
        RUN_MODE="mpi"
fi
echo "RUN_MODE: $RUN_MODE"

# now just reference the module
if [[ -z $LAUNCH_SCRIPT ]]; then
    LAUNCH_SCRIPT="spinifel"
fi
echo "LAUNCH_SCRIPT: $LAUNCH_SCRIPT"

t_start=`date +%s`

module load cudatoolkit

export IBV_FORK_SAFE=1
export RDMAV_HUGEPAGES_SAFE=1

SRUN_COMMAND="srun -N $NNODES --ntasks-per-node=$NTASKS_PER_NODE -c $NCPUS_PER_TASK --gpus $NGPUS --gpus-per-task=$NGPUS_PER_TASK python -m $LAUNCH_SCRIPT --default-settings=perlmutter_quickstart.toml --mode=$RUN_MODE runtime.N_images_per_rank=$N_IMAGES_PER_RANK algorithm.N_binning=$NBINNING algorithm.N_orientations=$NORIENT algorithm.N_batch_size=$BATCHSIZE data.out_dir=$OUT_DIR data.name=$DATA_FILENAME data.in_dir=$DATA_DIR"

echo "SRUN_COMMAND: $SRUN_COMMAND"

eval $SRUN_COMMAND

t_end=`date +%s`
echo SrunCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end
