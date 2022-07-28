#!/bin/bash
# Test tests orientation matching and mpi/main with xtc2 (no streaming).
# Environment variables explained:
#   test_data_dir: can be set to matchin with what stored permanently on 
#                  each SC. If not set, this script will try to download
#                  from testdata repo.
#   test_out_dir:  set by $target conditioning below
#   CI_PIPELINE_ID:If need to run this without the CI, you can create a
#                  folder under test_out_dir/${CI_PIPELINE_ID}/spinifel_output
#                  and set this to the chosen ID.


set -xe


## Clone test data and save them to test_data_dir
#if [ -z ${test_data_dir+x} ]; then
#    echo "[spinifel/test] - test_data_dir is not set. The testdata will be cloned"
#    module load git-lfs
#    git lfs install
#    test_data_root="/tmp" 
#    export test_data_dir="${test_data_root}/spinifel_testdata_$$"
#    echo "[spinifel/test] - Cloning testdata to $test_data_dir"
#    git clone https://gitlab.osti.gov/mtip/data/testdata.git $test_data_dir
#fi


target=${SPINIFEL_TARGET:-${NERSC_HOST:-$(hostname --fqdn)}}
echo "[spinifel/test] target: $target"
echo "[spinifel/test] test_data_dir: $test_data_dir"
echo "[spinifel/test] test_out_dir: $test_out_dir"
echo "[spinifel/test] CI_PIPELINE_ID: $CI_PIPELINE_ID"


if [[ ${target} = "cgpu"* ]]; then
    export test_data_dir="${CFS}/m2859/data/testdata"
    export test_out_dir="${CFS}/m2859/ci"

    export SPINIFEL_TEST_MODULE="ORIENTATION_MATCHING"
    srun -n 1 python -u -m spinifel --settings=./settings/test_mpi.toml --mode=test

    export SPINIFEL_TEST_MODULE="MAIN_PSANA2"
    srun -n 3 python -u -m spinifel --settings=./settings/test_mpi.toml --mode=test

elif [[ ${target} = *"summit"* || ${target} = *"ascent"* ]]; then
    if [[ ${target} = *"ascent"* ]]; then
	export all_proxy=socks://proxy.ccs.ornl.gov:3128/
	export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
	export http_proxy=http://proxy.ccs.ornl.gov:3128/
	export https_proxy=http://proxy.ccs.ornl.gov:3128/
	export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'
	export test_data_dir="/gpfs/wolf/chm137/proj-shared/testdata"
	export test_out_dir="/gpfs/wolf/chm137/proj-shared/ci"
    else
	export test_data_dir="/gpfs/alpine/proj-shared/chm137/data/testdata"
	export test_out_dir="/gpfs/alpine/proj-shared/chm137/ci"
    fi

    export SPINIFEL_TEST_MODULE="ORIENTATION_MATCHING"
    jsrun -n1 -g1 python -u -m spinifel --settings=./settings/test_mpi.toml --mode=test

    export SPINIFEL_TEST_MODULE="MAIN_PSANA2"
    jsrun -n3 -g1 python -u -m spinifel --settings=./settings/test_mpi.toml --mode=test
fi
