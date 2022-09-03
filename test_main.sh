#!/bin/bash


set -xe

export CI_PIPELINE_ID=000
export SPINIFEL_TEST_LAUNCHER="jsrun -n1 -a1 -g1"

#$SPINIFEL_TEST_LAUNCHER python -m spinifel --default-settings=summit_ci.toml --mode=mpi runtime.use_cufinufft=false

#PYTHONPATH="$PYTHONPATH:$EXTERNAL_WORKDIR:$PWD/mpi4py_poison_wrapper" $SPINIFEL_TEST_LAUNCHER legion_python -ll:py 1 -ll:csize 8192 legion_main.py --default-settings=summit_ci.toml --mode=legion




export SPINIFEL_TEST_MODULE="MAIN_PSANA2"
export SPINIFEL_TEST_LAUNCHER="jsrun -n3 -g1"
target=${SPINIFEL_TARGET:-${NERSC_HOST:-$(hostname --fqdn)}}
export PYCUDA_CACHE_DIR="/tmp"
if [[ ${target} = *"ascent"* ]]; then
    export all_proxy=socks://proxy.ccs.ornl.gov:3128/
    export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
    export http_proxy=http://proxy.ccs.ornl.gov:3128/
    export https_proxy=http://proxy.ccs.ornl.gov:3128/
    export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'
    export test_data_dir="/gpfs/wolf/chm137/proj-shared/spinifel_data/testdata"
    export OUT_DIR="/gpfs/wolf/chm137/proj-shared/ci/${CI_PIPELINE_ID}/spinifel_output"
else
    export test_data_dir="/gpfs/alpine/proj-shared/chm137/data/testdata"
    export test_out_dir="/gpfs/alpine/proj-shared/chm137/ci"
fi
$SPINIFEL_TEST_LAUNCHER python -u -m spinifel --settings=./settings/test_mpi.toml --mode=mpi

