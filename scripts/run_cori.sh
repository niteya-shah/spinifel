#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --qos=premium
#SBATCH --constraint=haswell
#SBATCH --mail-type=ALL
#SBATCH --account=m2859

while getopts lpn: option
do
case "${option}"
in
l) USING_LEGION="1";;
p) PROFILING="1";;
n) NTASKS=$OPTARG;;
esac
done

root_dir="$PWD"

source "$root_dir"/setup/env.sh

export PYTHONPATH="$PYTHONPATH:$root_dir"

export DATA_DIR=$SCRATCH/spinifel_data

export OUT_DIR=$SCRATCH/spinifel_output
mkdir -p $OUT_DIR
rm -rf $OUT_DIR/*

export OMP_NUM_THREADS=1
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"

nodes=$SLURM_JOB_NUM_NODES

if [[ -z $NTASKS ]]; then
	NTASKS="1"
fi
echo "NTASKS: $NTASKS"

if [[ $USING_LEGION -eq 1 ]]; then
    sockets=2
    cores=10
    srun -n $NTASKS legion_python legion_main.py -ll:py 1 -ll:csize 16384
else
    if [[ $PROFILING -eq 1 ]]; then
        PYFLAGS="-m cProfile -o $OUT_DIR/main.prof "$PYFLAGS
    fi
    python $PYFLAGS sequential_main.py
fi
