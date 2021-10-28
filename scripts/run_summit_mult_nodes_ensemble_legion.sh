#!/bin/bash
#set -x
#old
#while getopts msca:t:d:g:n:fr:el: option
while getopts ma:t:c:i:g:n:r:el:o:b:s: option
do
case "${option}"
in
a) NTASKS_PER_RS=$OPTARG;;
c) CPUS_PER_RS=$OPTARG;;
#using this?
t) OMP_NUM_THREADS=$OPTARG;;
i) N_IMAGES_PER_RANK=$OPTARG;;
g) DEVICES_PER_RS=$OPTARG;;
n) NRESOURCESETS=$OPTARG;;
r) NRSS_PER_NODE=$OPTARG;;
e) CHECK_FOR_ERRORS="1";;
l) LAUNCH_SCRIPT=$OPTARG;;
o) NORIENT=$OPTARG;;
b) NBINNING=$OPTARG;;
s) BATCHSIZE=$OPTARG;;
esac
done

if [[ -n CHECK_FOR_ERRORS ]]; then
    echo "CHECK_FOR_ERRORS: $CHECK_FOR_ERRORS"
    set -e
fi

# # i guess keep this
# export LD_PRELOAD=/sw/summit/gcc/8.1.1-cuda10.1.168/lib64/libgomp.so.1

root_dir=${root_dir:-"$PWD"}
echo "root_dir: $root_dir"

source "$root_dir"/setup/env.sh

#assume we keep these
# export PYTHONPATH="$PYTHONPATH:$root_dir"
export MPLCONFIGDIR=/tmp

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
# export DEVICES_PER_RS
echo "DEVICES_PER_RS: $DEVICES_PER_RS"

if [[ -z $NRSS_PER_NODE ]]; then
        NRSS_PER_NODE="1"
fi
echo "NRSS_PER_NODE: $NRSS_PER_NODE"

# # used by some libraries?
# if [[ -n $OMP_NUM_THREADS ]]; then
#     export OMP_NUM_THREADS
#     echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
# fi


# export RUN_MODE=legion
# echo "RUN_MODE: $RUN_MODE"

#for MPI
#export SPINIFEL_PYTHON="python"
#for legion
SPINIFEL_PYTHON="legion_python -ll:py 1 -ll:csize 8192 -ll:show_rsrv"
#export SPINIFEL_PYTHON="legion_python -ll:py 1 -ll:csize 8192 -ll:show_rsrv"
#for MPI
#export LAUNCH_COMMAND="-m spinifel"
#for legion
LAUNCH_COMMAND=legion_main.py
export CUPY_CACHE_DIR=/tmp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/alpine/chm137/scratch/cahrens/devel/spinifel-devel-092921/setup/conda/pkgs/openjpeg-2.4.0-hfe35807_0/lib

JSRUN_COMMAND="jsrun -n $NRESOURCESETS -a $NTASKS_PER_RS -c $CPUS_PER_RS -g $DEVICES_PER_RS -r $NRSS_PER_NODE -b rs -d packed $SPINIFEL_PYTHON $LAUNCH_COMMAND --default-settings=summit_quickstart.toml --mode=$RUN_MODE runtime.N_images_per_rank=$N_IMAGES_PER_RANK algorithm.N_binning=$NBINNING algorithm.N_orientations=$NORIENT algorithm.N_batch_size=$BATCHSIZE data.out_dir=$OUT_DIR data.name=$DATA_FILENAME data.in_dir=$DATA_DIR"

#env &> env.out

echo "JSRUN_COMMAND: $JSRUN_COMMAND"

set -x

$JSRUN_COMMAND
