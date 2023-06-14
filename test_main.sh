#!/bin/bash
#####################################################################
######## Run all tests specified in main test on CI #################
#####################################################################

set -xe


export CI_PIPELINE_ID=000
target=${SPINIFEL_TARGET:-${NERSC_HOST:-$(hostname --fqdn)}}
export SPINIFEL_TEST_FLAG=1


export PYCUDA_CACHE_DIR="/tmp"


# Set internet proxy for summit and ascent. Psana2 needs this access.
if [[ ${target} = *"summit"* || ${target} = *"ascent"* || ${target} = *"crusher"* || ${target} = *"frontier"* ]]; then
    export all_proxy=socks://proxy.ccs.ornl.gov:3128/
    export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
    export http_proxy=http://proxy.ccs.ornl.gov:3128/
    export https_proxy=http://proxy.ccs.ornl.gov:3128/
    export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'
fi


# Set input and output folders
if [[ ${target} = *"ascent"* ]]; then
    export test_data_dir="/gpfs/wolf/chm137/proj-shared/spinifel_data/testdata"
    export out_dir="/gpfs/wolf/chm137/proj-shared/ci/${CI_PIPELINE_ID}/spinifel_output"
elif [[ ${target} = *"summit"* ]]; then
    export test_data_dir="/gpfs/alpine/proj-shared/chm137/data/testdata"
    export out_dir="/gpfs/alpine/proj-shared/chm137/test_main/${CI_PIPELINE_ID}/spinifel_output"
elif [[ ${target} = *"frontier"* ]]; then
    export test_data_dir="/lustre/orion/chm137/proj-shared/testdata"
    export out_dir="/lustre/orion/chm137/scratch/${USER}/${CI_PIPELINE_ID}/spinifel_output"
elif [[ ${target} = *"crusher"* ]]; then
    export test_data_dir="/lustre/orion/chm137/proj-shared/testdata"
    export out_dir="/lustre/orion/chm137/scratch/${USER}/${CI_PIPELINE_ID}_crusher/spinifel_output"
elif [[ ${target} = *"cgpu"* || ${target} = *"perlmutter"* ]]; then
    export test_data_dir="${CFS}/m2859/data/testdata"
    export out_dir="${SCRATCH}/spinifel_output"
fi


# Set job submisson command
if [[ ${target} = *"summit"* || ${target} = *"ascent"* ]]; then
    export SPINIFEL_TEST_LAUNCHER="jsrun -n1 -a1 -g1"
    export SPINIFEL_LEGION_LAUNCHER="jsrun -n1 -a1 -g1"
    export SPINIFEL_PSANA2_LAUNCHER="jsrun -n3 -g1"
    export SPINIFEL_TEST_LAUNCHER_MULTI="jsrun -n2 -a1 -g1"
elif [[ ${target} = *"cgpu"* || ${target} = *"perlmutter"* || ${target} = *"frontier"* || ${target} = *"crusher"* ]]; then
    export SPINIFEL_TEST_LAUNCHER="srun -n1 -G1"
    export SPINIFEL_LEGION_LAUNCHER="srun -N2 -n2 -G2"
    export SPINIFEL_PSANA2_LAUNCHER="srun -n3 -G3"
    export SPINIFEL_TEST_LAUNCHER_MULTI="srun -n2 --gpus-per-task 1"
fi


# Creates the output folder if not already exist.
if [ ! -d "${out_dir}" ]; then
    mkdir -p ${out_dir}
fi


# Use PybindGPU instead of PyCUDA
if [[ ${target} = *"frontier"* || ${target} = *"crusher"* ]]; then
    FRONTIER_EXTRAS="runtime.use_pygpu=true"
fi


# spinifel legion
export SPINIFEL_LEGION="legion_python -ll:py 1  -ll:pyomp 2 -ll:csize 32768 legion_main.py  --default-settings=test_mpi.toml ${FRONTIER_EXTRAS}"


#DEBUG_FLAG="-Xfaulthandler"

export USE_CUPY=1

export USE_CUPY=1


# unittest
pytest -s spinifel/tests/test_main.py


# test_mpi_hdf5
$SPINIFEL_TEST_LAUNCHER python $DEBUG_FLAG -m spinifel --default-settings=test_mpi.toml --mode=mpi $FRONTIER_EXTRAS


# test_legion_hdf5_multirank
PYTHONPATH="$PYTHONPATH:$EXTERNAL_WORKDIR:$PWD/mpi4py_poison_wrapper" $SPINIFEL_TEST_LAUNCHER_MULTI $SPINIFEL_LEGION --mode=legion



# test_mpi_xtc2
$SPINIFEL_PSANA2_LAUNCHER python -u -m spinifel --default-settings=test_mpi.toml --mode=mpi psana.enable=true $FRONTIER_EXTRAS


# test_legion_xtc2_stream_multirank
PYTHONPATH="$PYTHONPATH:$EXTERNAL_WORKDIR:$PWD/mpi4py_poison_wrapper" $SPINIFEL_TEST_LAUNCHER_MULTI $SPINIFEL_LEGION --mode=legion_psana2 psana.enable=true psana.ps_smd_n_events=1000 algorithm.N_gens_stream=2 algorithm.N_images_max=4000


# test_finufft
$SPINIFEL_TEST_LAUNCHER python -m spinifel --default-settings=test_mpi.toml --mode=mpi runtime.use_cuda=false runtime.use_cufinufft=false fsc.fsc_min_cc=0.6 fsc.fsc_min_change_cc=0.1 runtime.use_single_prec=false $FRONTIER_EXTRAS


# test_nocuda
$SPINIFEL_TEST_LAUNCHER python -m spinifel --default-settings=test_mpi.toml --mode=mpi runtime.use_cufinufft=false runtime.use_cuda=false runtime.use_cupy=false fsc.fsc_min_cc=0.6 fsc.fsc_min_change_cc=0.1 runtime.use_single_prec=false $FRONTIER_EXTRAS




