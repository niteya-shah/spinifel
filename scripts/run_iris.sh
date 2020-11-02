#!/bin/bash
#PBS -n 1
#PBS -t 30
#PBS -q iris

#cd $PBS_O_WORKDIR

while getopts lmpsavn:t:d: option
do
case "${option}"
in
l) USING_LEGION="1";;
m) USING_MPI="1";;
p) PROFILING="1";;
s) SMALL_PROBLEM="1";;
a) USE_PSANA="1";;
v) VERBOSITY="1";;
n) NTASKS=$OPTARG;;
t) OMP_NUM_THREADS=$OPTARG;;
d) DATA_MULTIPLIER=$OPTARG;;
esac
done

if [[ $USING_LEGION -eq 1 && $USING_MPI -eq 1 ]]; then
	echo "Legion and MPI options are mutually exclusive. Please pick one."
	exit 1
fi

root_dir="$PWD"

source "$root_dir"/setup/env.sh

export PYTHONPATH="$PYTHONPATH:$root_dir"

export DATA_DIR=/gpfs/jlse-fs0/users/$USER/spinifel_data

export OUT_DIR=/gpfs/jlse-fs0/users/$USER/spinifel_output
mkdir -p $OUT_DIR
rm -rf $OUT_DIR/*

nodes=1 #$SLURM_JOB_NUM_NODES
total_cores=10 #$(( $(echo $SLURM_JOB_CPUS_PER_NODE | cut -d'(' -f 1) / 2 ))

if [[ -z $NTASKS ]]; then
	NTASKS="1"
fi
echo "NTASKS: $NTASKS"

if [[ -n $OMP_NUM_THREADS ]]; then
    export OMP_NUM_THREADS
    echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
fi

if [[ -n $SMALL_PROBLEM ]]; then
    export SMALL_PROBLEM
fi

if [[ -n $USE_PSANA ]]; then
    export USE_PSANA
fi

if [[ -n $VERBOSITY ]]; then
    export VERBOSITY
fi

if [[ -z $DATA_MULTIPLIER ]]; then
    DATA_MULTIPLIER="1"
fi
export DATA_MULTIPLIER

if [[ $USING_LEGION -eq 1 ]]; then
    export PS_PARALLEL=legion
    srun -n $NTASKS -N $nodes --cpus-per-task=$(( total_cores * 2 / (NTASKS / nodes) )) legion_python legion_main.py -ll:csize 65536 -ll:py 1 -ll:pyomp $(( total_cores - 2 )) $LGFLAGS
elif [[ $USING_MPI -eq 1 ]]; then
    export PS_PARALLEL=mpi
    srun -n $NTASKS python mpi_main.py
else
    export PS_PARALLEL=none
    if [[ $PROFILING -eq 1 ]]; then
        PYFLAGS="-m cProfile -o $OUT_DIR/main.prof "$PYFLAGS
    fi
    DATA_MULTIPLIER=$(echo $(( DATA_MULTIPLIER * NTASKS )) )  # use same amount of data as distributed app
    python $PYFLAGS sequential_main.py
fi
