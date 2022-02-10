#!/bin/bash
t_start=`date +%s`

#print time with timezone and trial name
export TRIAL=clean_3iyf_ensemble_perf
export RUNDATE=`date +"%FT%H%M%z"`
export OUT_DIR_BASE=${SCRATCH}/spinifel_output/spinifel_pm_${RUNDATE}_TRIAL_${TRIAL}
mkdir -p $OUT_DIR_BASE
export DATA_DIR=${CFS}/m2859/data/3iyf/clean
export DATA_FILENAME=3iyf_128x128pixels_500k.h5
export VERBOSE=true

#N) NNODES=$OPTARG;;
#n) NTASKS_PER_NODE=$OPTARG;;
#c) NCPUS_PER_TASK=$OPTARG;;
#g) NGPUS_PER_NODE=$OPTARG;;
#f) NGPUS_PER_TASK=$OPTARG;;
#m) RUN_MODE=$OPTARG;;
#i) N_IMAGES_PER_RANK=$OPTARG;;
#o) NORIENT=$OPTARG;;
#b) NBINNING=$OPTARG;;
#s) BATCHSIZE=$OPTARG;;

for NODES in 4
do
for NIMAGES in 1000 #500 
do
for NORIENT in 3000 #10000 
do
for NBINNING in 0
do
for BATCHSIZE in 100
do

export PARAMS=nodes_${NODES}_nimages_${NIMAGES}_norient_${NORIENT}_nbinning_${NBINNING}_nbatchsize_${BATCHSIZE}
export OUT_DIR=${OUT_DIR_BASE}/nodes_${NODES}_nimages_${NIMAGES}_norient_${NORIENT}_nbinning_${NBINNING}_nbatchsize_${BATCHSIZE}
mkdir -p $OUT_DIR
export JOB_NAME=RunSpinifel_${TRIAL}_${PARAMS}
export ERR_LOG=${OUT_DIR_BASE}/error_%J_${PARAMS}.log
export OUT_LOG=${OUT_DIR_BASE}/output_%J_${PARAMS}.log

export IBV_FORK_SAFE=1
export RDMAV_HUGEPAGES_SAFE=1

sbatch -v -A m3890_g -C gpu -q early_science -t 1:00:00 -N $NODES --tasks-per-node=4 -c 4 --gpus 16 --gpus-per-task=1  -J ${JOB_NAME} -e ${ERR_LOG} -o ${OUT_LOG} --wrap="./scripts/run_pm.sh -N $NODES -n 4 -c 4 -g 16 -f 1 -m mpi -i $NIMAGES -o $NORIENT -b $NBINNING -s $BATCHSIZE"

done
done
done
done
done

t_end=`date +%s`
echo BatchJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end


