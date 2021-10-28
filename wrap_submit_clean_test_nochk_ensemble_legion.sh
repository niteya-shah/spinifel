#!/bin/bash

#set -x

#need this first so can tell bsub where to put stuff

#print time with timezone and trial name
export TRIAL=clean_test_3iyf_nochk_0929devel_10korient_legion
export RUNDATE=`date +"%FT%H%M%z"`
export OUT_DIR_BASE=$MEMBERWORK/chm137/spinifel_output_${RUNDATE}_TRIAL_${TRIAL}
mkdir -p $OUT_DIR_BASE
export ERR_LOG=${OUT_DIR_BASE}/error.%J.log
export OUT_LOG=${OUT_DIR_BASE}/output.%J.log
export DATA_DIR=/gpfs/alpine/world-shared/chm137/iris/
export DATA_FILENAME=3iyf_128x128pixels_500k.h5
#export DATA_DIR=/gpfs/alpine/chm137/proj-shared/data/spi
#export DATA_FILENAME=2CEX-10k-2.h5 
export VERBOSE=true

# Note:
# -n is number resource sets
# -r is resource sets per node
# -a is tasks per resource set
# -c is CPUs per resource set
# -g is devices per resource set

for NODES in 1 
do
for RS_PER_NODE in 2 
do
export NUM_RES_SET=$((NODES*RS_PER_NODE))
export NUM_CPUS_PER_RS=7
export TASKS_PER_RS=1
for NIMAGES in 1000
do
for NORIENT in 10000
do
for NBINNING in 0
do
for BATCHSIZE in 100
do
OUT_SUBDIR=${OUT_DIR_BASE}/nodes_${NODES}_rspernode_${RS_PER_NODE}_nimages_${NIMAGES}_norient_${NORIENT}_nbinning_${NBINNING}_nbatchsize_${BATCHSIZE}
mkdir -p $OUT_SUBDIR
export OUT_DIR=$OUT_SUBDIR
export CHK_CONVERGENCE=false

set -x

bsub -P CHM137 -J RunSpinifel_${TRIAL}_nodes_${NODES}_rspernode_${RS_PER_NODE}_nimages_${NIMAGES}_norient_${NORIENT}_nbinning_${NBINNING}_nbatchsize_${BATCHSIZE} -W 2:00 -nnodes ${NODES} -e ${ERR_LOG}_nodes_${NODES}_rspernode_${RS_PER_NODE}_nimages_${NIMAGES}_norient_${NORIENT}_nbinning_${NBINNING}_nbatchsize_${BATCHSIZE} -o ${OUT_LOG}_nodes_${NODES}_rspernode_${RS_PER_NODE}_nimages_${NIMAGES}_norient_${NORIENT}_nbinning_${NBINNING}_nbatchsize_${BATCHSIZE} -alloc_flags "gpumps" "bash scripts/run_summit_mult_nodes_ensemble_legion.sh -n $NUM_RES_SET -r $RS_PER_NODE -a $TASKS_PER_RS -c $NUM_CPUS_PER_RS -i $NIMAGES -g 1 -o $NORIENT -b $NBINNING -s $BATCHSIZE"
done
done
done
done
done
done
