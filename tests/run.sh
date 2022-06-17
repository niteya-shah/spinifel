#!/bin/bash

set -xe

# Clone test data and save them to test_data_dir
if [ -z ${test_data_dir+x} ]; then
    echo "[spinifel/test] - test_data_dir is not set. The testdata will be cloned"
    test_data_root="/tmp" 
    export test_data_dir="${test_data_root}/spinifel_testdata_$$"
    echo "[spinifel/test] - Cloning testdata to $test_data_dir"
    git clone https://gitlab.osti.gov/mtip/data/testdata.git $test_data_dir
fi


target=`hostname`


if [[ ${target} = "cgpu"* ]]; then
    export SPINIFEL_TEST_MODULE="ORIENTATION_MATCHING"
    srun -n 1 python -u -m spinifel --settings=./settings/test_mpi.toml --mode=test

    export SPINIFEL_TEST_MODULE="MAIN_PSANA2"
    srun -n 3 python -u -m spinifel --settings=./settings/test_mpi.toml --mode=test

elif [[ ${target} = *"summit"* ]]; then
    export SPINIFEL_TEST_MODULE="ORIENTATION_MATCHING"
    jsrun -n 1 python -u -m spinifel --settings=./settings/test_mpi.toml --mode=test

    export SPINIFEL_TEST_MODULE="MAIN_PSANA2"
    jsrun -n 3 python -u -m spinifel --settings=./settings/test_mpi.toml --mode=test
fi
