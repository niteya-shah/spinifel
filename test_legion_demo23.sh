#!/bin/bash -x
export OUT_DIR_LEGION="/lustre/orion/proj-shared/chm137/${USER}/spinifel_output_frontier_opt"
export USE_CUPY=1
export test_data_dir="/lustre/orion/proj-shared/chm137/"
export OPTIONS="legion_python -ll:pyomp 4 -ll:py 1 -level runtime=5 -ll:csize 8192 -gex:amlimit 32 legion_main.py --default-settings=multiple_conform.toml --mode=legion runtime.N_images_per_rank=1000 algorithm.N_generations=10 runtime.chk_convergence=false runtime.cupy_mempool_clear=false debug.verbosity=0 algorithm.N_conformations=2 algorithm.conformation_mode=max_likelihood algorithm.max_intensity_clip=2.5 debug.show_image=false"
t_start=`date +%s`
source setup/env.sh
export LEGION_BACKTRACE=1
export REALM_BACKTRACE=1
export PS_PARALLEL='legion'
export PS_SMD_N_EVENTS=1000
export USE_CUPY=1
export FI_MR_CACHE_MONITOR=memhooks
export FI_CXI_RX_MATCH_MODE=software
export GASNET_OFI_DEVICE_0=cxi2
export GASNET_OFI_DEVICE_1=cxi1
export GASNET_OFI_DEVICE_2=cxi3
export GASNET_OFI_DEVICE_3=cxi0
export GASNET_OFI_DEVICE_TYPE=Node
export GASNET_OFI_RECEIVE_BUFF_SIZE=single
default_options=""
options=${OPTIONS:="${default_options}"}
OUT_DIR_LEGION=${OUT_DIR_LEGION:="${SCRATCH}/spinifel_output_legion"}
echo $OUT_DIR_LEGION
echo $options
export out_dir="${OUT_DIR_LEGION}"
mkdir -p $OUT_DIR_LEGION
ntasks_per_node=8
echo 'Running Spinifel  Legion with Multiple Conformations'
for n in $SLURM_JOB_NUM_NODES ; do
    ntasks=$(( n*8 ))
    echo "Ranks: "$ntasks" "
    echo "Options: " $options" "
    OUT_DIR_LEGION="${OUT_DIR_LEGION}/result_${ntasks}tasks"
    echo "out_dir: " $OUT_DIR_LEGION" "
    mkdir -p $OUT_DIR_LEGION
    srun -N $n  -n $ntasks -c7 --gpus-per-task 1 --gpu-bind=single:1 $options runtime.use_pygpu=true data.out_dir="${OUT_DIR_LEGION}" 
    echo 'Finished Legion'
t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end
done
