#!/bin/bash
#BSUB -P CHM137
#BSUB -J fxs                  # job name
#BSUB -W 02:00                # wall-clock time (hrs:mins)
#BSUB -nnodes 1               # number of tasks in job
#BSUB -alloc_flags "gpumps"
#BSUB -e error.%J.log         # error file name in which %J is replaced by the job ID
#BSUB -o output.%J.log        # output file name in which %J is replaced by the job ID


while getopts msca:t:d:g:n:fr:e option
do
case "${option}"
in
m) USING_MPI="1";;
s) SMALL_PROBLEM="1";;
c) USING_CUDA="1";;
a) NTASKS_PER_RS=$OPTARG;;
t) OMP_NUM_THREADS=$OPTARG;;
d) DATA_MULTIPLIER=$OPTARG;;
g) DEVICES_PER_RS=$OPTARG;;
n) NRESOURCESETS=$OPTARG;;
f) USE_CUFINUFFT="1";;
r) NRSS_PER_NODE=$OPTARG;;
e) CHECK_FOR_ERRORS="1";;
esac
done

if [[ -n CHECK_FOR_ERRORS ]]; then
    echo "CHECK_FOR_ERRORS: $CHECK_FOR_ERRORS"
    set -e
fi

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

export DATA_DIR=${DATA_DIR:-/gpfs/alpine/proj-shared/chm137/data/testdata/2CEX}
export DATA_FILENAME=${DATA_FILENAME:-2CEX-10.h5}

export OUT_DIR=${OUT_DIR:-/gpfs/alpine/proj-shared/chm137/$USER/spinifel_output}
mkdir -p $OUT_DIR
rm -rf $OUT_DIR/*

if [[ -z $NRESOURCESETS ]]; then
        NRESOURCESETS="1"
fi
echo "NRESOURCESETS: $NRESOURCESETS"

if [[ -z $NTASKS_PER_RS ]]; then
        NTASKS_PER_RS="1"
fi
echo "NTASKS_PER_RS: $NTASKS_PER_RS"

if [[ -z $DEVICES_PER_RS ]]; then
    DEVICES_PER_RS="1"
fi
export DEVICES_PER_RS
echo "DEVICES_PER_RS: $DEVICES_PER_RS"

if [[ -z $NRSS_PER_NODE ]]; then
        NRSS_PER_NODE="1"
fi
echo "NRSS_PER_NODE: $NRSS_PER_NODE"

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
    cd "$root_dir"/spinifel/sequential/
    nvcc -O3 -shared -std=c++11 `python3 -m pybind11 --includes` orientation_matching.cu -o pyCudaKNearestNeighbors`python3-config --extension-suffix`
    cd "$root_dir"
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


echo "MPI run"
export PS_PARALLEL=mpi
export VERBOSE=true

# TO RUN THE UNIT TEST FOR ORIENTATION MATCHING
# Replace finufftpy with finufft
#USE_ORIGINAL_FINUFFT=1
#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/ccs/home/monarin/sw/spinifel/setup/finufft_original/lib"
#export PYTHONPATH="$PYTHONPATH:/ccs/home/monarin/sw/spinifel/setup/finufft_original/python"

#jsrun -n 1 -g 1 python spinifel/tests/test_orientation_matching.py

#export DEBUG_FLAG=1
jsrun -n $NRESOURCESETS -a $NTASKS_PER_RS -c $NTASKS_PER_RS -g $DEVICES_PER_RS -r $NRSS_PER_NODE python mpi_main.py

#jsrun -n $NRESOURCESETS -a $NTASKS_PER_RS -c $NTASKS_PER_RS -g $DEVICES_PER_RS gdb python -x scripts/run_gdb
