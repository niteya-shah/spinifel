#!/bin/bash
#BSUB -P CHM137
#BSUB -J fxs                  # job name
#BSUB -W 02:00                # wall-clock time (hrs:mins)
#BSUB -nnodes 1               # number of tasks in job
#BSUB -alloc_flags "gpumps"
#BSUB -e error.%J.log         # error file name in which %J is replaced by the job ID
#BSUB -o output.%J.log        # output file name in which %J is replaced by the job ID

while getopts mscn:t:d:g:N:f option
do
case "${option}"
in
m) USING_MPI="1";;
s) SMALL_PROBLEM="1";;
c) USING_CUDA="1";;
n) NTASKS_PER_NODE=$OPTARG;;
t) OMP_NUM_THREADS=$OPTARG;;
d) DATA_MULTIPLIER=$OPTARG;;
g) DEVICES_PER_NODE=$OPTARG;;
N) NNODES=$OPTARG;;
f) USE_CUFINUFFT="1";;
esac
done

export LD_PRELOAD=/sw/summit/gcc/8.1.1-cuda10.1.168/lib64/libgomp.so.1
if [[ -n $USING_MPI ]]; then
    export USING_MPI
    echo "USING_MPI: $USING_MPI"
fi

root_dir="$PWD"
echo "root_dir: $root_dir"

source "$root_dir"/setup/env.sh


export PYTHONPATH="$PYTHONPATH:$root_dir"
export MPLCONFIGDIR=/gpfs/alpine/scratch/$USER/chm137/mtipProxy/writableDirectory

#export DATA_DIR=$SCRATCH/spinifel_data
export DATA_DIR=/gpfs/alpine/proj-shared/chm137/data/spi
#export DATA_DIR=/gpfs/alpine/proj-shared/chm137/iris

#export OUT_DIR=$SCRATCH/spinifel_output
export OUT_DIR=/gpfs/alpine/proj-shared/chm137/blaschke/spinifel_output
mkdir -p $OUT_DIR
rm -rf $OUT_DIR/*

if [[ -z $NNODES ]]; then
        NNODES="1"
fi
echo "NNODES: $NNODES"

if [[ -z $NTASKS_PER_NODE ]]; then
        NTASKS_PER_NODE="1"
fi
echo "NTASKS_PER_NODE: $NTASKS_PER_NODE"

if [[ -n $OMP_NUM_THREADS ]]; then
    export OMP_NUM_THREADS
    echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
fi

if [[ -n $SMALL_PROBLEM ]]; then
    export SMALL_PROBLEM
    echo "SMALL_PROBLEM: $SMALL_PROBLEM"
fi

if [[ -n $USING_CUDA ]]; then
    export USING_CUDA
    echo "CUDA: $USING_CUDA"
    #cd "$root_dir"/spinifel/sequential/
    #nvcc -O3 -shared -std=c++11 `python3 -m pybind11 --includes` orientation_matching.cu -o pyCudaKNearestNeighbors`python3-config --extension-suffix`
    #cd "$root_dir"
fi

if [[ -n $USE_PSANA ]]; then
    export USE_PSANA
fi

if [[ -n $USE_CUFINUFFT ]]; then
    export USE_CUFINUFFT
fi
echo "USE_CUFINUFFT: $USE_CUFINUFFT"

if [[ -z $DATA_MULTIPLIER ]]; then
    DATA_MULTIPLIER="1"
fi
export DATA_MULTIPLIER
echo "DATA_MULTIPLIER: $DATA_MULTIPLIER"

if [[ -z $DEVICES_PER_NODE ]]; then
    DEVICES_PER_NODE="6"
fi
export DEVICES_PER_NODE
echo "DEVICES_PER_NODE: $DEVICES_PER_NODE"

echo "MPI run"
export PS_PARALLEL=mpi
export VERBOSE=true
#jsrun -n 1 -a 42 -c 42 -r 1 -g 6 python mpi_main.py
jsrun -n $NNODES -a $NTASKS_PER_NODE -c $NTASKS_PER_NODE -g $DEVICES_PER_NODE python mpi_main.py
