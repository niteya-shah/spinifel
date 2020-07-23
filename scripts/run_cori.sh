#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --qos=premium
#SBATCH --constraint=haswell
#SBATCH --mail-type=ALL
#SBATCH --account=m2859

while getopts lmpsn:t: option
do
case "${option}"
in
l) USING_LEGION="1";;
m) USING_MPI="1";;
p) PROFILING="1";;
s) SMALL_PROBLEM="1";;
n) NTASKS=$OPTARG;;
t) OMP_NUM_THREADS=$OPTARG;;
esac
done

if [[ $USING_LEGION -eq 1 && $USING_MPI -eq 1 ]]; then
	echo "Legion and MPI options are mutually exclusive. Please pick one."
	exit 1
fi

root_dir="$PWD"

source "$root_dir"/setup/env.sh

export PYTHONPATH="$PYTHONPATH:$root_dir"

export DATA_DIR=$SCRATCH/spinifel_data

export OUT_DIR=$SCRATCH/spinifel_output
mkdir -p $OUT_DIR
rm -rf $OUT_DIR/*

nodes=$SLURM_JOB_NUM_NODES
total_cores=$(( $(echo $SLURM_JOB_CPUS_PER_NODE | cut -d'(' -f 1) / 2 ))

if [[ -z $NTASKS ]]; then
	NTASKS="1"
fi
echo "NTASKS: $NTASKS"

if [[ -n $OMP_NUM_THREADS ]]; then
    export OMP_NUM_THREADS
    echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
fi

export SMALL_PROBLEM

if [[ $USING_LEGION -eq 1 ]]; then
    srun -n $NTASKS -N $nodes --cpus-per-task=$(( total_cores * 2 / (NTASKS / nodes) )) legion_python legion_main.py -ll:csize 16384 -ll:py 1 -ll:pyomp $(( total_cores - 2 ))
elif [[ $USING_MPI -eq 1 ]]; then
    srun -n $NTASKS python mpi_main.py
else
    if [[ $PROFILING -eq 1 ]]; then
        PYFLAGS="-m cProfile -o $OUT_DIR/main.prof "$PYFLAGS
    fi
    export DATA_MULTIPLIER=$NTASKS  # use same amount of data as distributed app
    python $PYFLAGS sequential_main.py
fi
