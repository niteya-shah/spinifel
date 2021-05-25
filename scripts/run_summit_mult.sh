#!/bin/bash
#BSUB -P CHM137
#BSUB -J fxs                  # job name
#BSUB -W 02:00                # wall-clock time (hrs:mins)
#BSUB -nnodes 1               # number of tasks in job
#BSUB -alloc_flags "gpumps"
#BSUB -e error.%J.log         # error file name in which %J is replaced by the job ID
#BSUB -o output.%J.log        # output file name in which %J is replaced by the job ID


while getopts mLsca:t:d:g:n:fr:el:P: option
do
case "${option}"
in
m) USING_MPI="1";;
L) USING_LEGION="1";;
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
l) LAUNCH_SCRIPT=$OPTARG;;
P) PROFILE=$OPTARG;;
esac
done

if [[ -n CHECK_FOR_ERRORS ]]; then
    echo "CHECK_FOR_ERRORS: $CHECK_FOR_ERRORS"
    set -e
fi

export LD_PRELOAD=/sw/summit/gcc/8.1.1-cuda10.1.168/lib64/libgomp.so.1
if [[ -n $USING_MPI && -n $USING_LEGION ]]; then
    echo "MPI and Legion options are mutually exclusive. Please pick one."
    exit 1
fi
if [[ -n $USING_MPI ]]; then
    export USING_MPI
    echo "USING_MPI: $USING_MPI"
elif [[ -n $USING_LEGION ]]; then
    export USING_LEGION
    echo "USING_LEGION: $USING_LEGION"
else
    echo "No runtime system was specified. Please choose either MPI or Legion."
    exit 1
fi

root_dir=${root_dir:-"$PWD"}
echo "root_dir: $root_dir"

source "$root_dir"/setup/env.sh


export PYTHONPATH="$PYTHONPATH:$root_dir"
export MPLCONFIGDIR=/gpfs/alpine/scratch/$USER/chm137/mtipProxy/writableDirectory

export DATA_DIR=${DATA_DIR:-/gpfs/alpine/proj-shared/chm137/data/spi}
export DATA_FILENAME=${DATA_FILENAME:-2CEX-10k-2.h5}

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

if [[ -z $LAUNCH_SCRIPT ]]; then
    if [[ -n $USING_MPI ]]; then
        LAUNCH_SCRIPT=(python "$root_dir/mpi_main.py")
    elif [[ -n $USING_LEGION ]]; then
        LAUNCH_SCRIPT=(legion_python "$root_dir/legion_main.py" -ll:py 1 -ll:csize 8192)
    fi
else
    LAUNCH_SCRIPT=(python "$LAUNCH_SCRIPT")
fi

if [[ -n $PROFILE ]]; then
    LAUNCH_SCRIPT=(nsys profile -o ${PROFILE}.%q{OMPI_COMM_WORLD_RANK} -f true --stats=true ${LAUNCH_SCRIPT[@]})
fi


echo "LAUNCH_SCRIPT: $LAUNCH_SCRIPT"

if [[ -n $USING_MPI ]]; then
    echo "MPI run"
    export PS_PARALLEL=mpi
elif [[ -n $USING_LEGION ]]; then
    echo "Legion run"
    export PS_PARALLEL=legion
fi
export VERBOSE=true

# TO RUN THE UNIT TEST FOR ORIENTATION MATCHING
# Replace finufftpy with finufft
#USE_ORIGINAL_FINUFFT=1
#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/ccs/home/monarin/sw/spinifel/setup/finufft_original/lib"
#export PYTHONPATH="$PYTHONPATH:/ccs/home/monarin/sw/spinifel/setup/finufft_original/python"

#export DEBUG_FLAG=1
set -x
jsrun -n $NRESOURCESETS -a $NTASKS_PER_RS -c $NTASKS_PER_RS -g $DEVICES_PER_RS "${LAUNCH_SCRIPT[@]}"

