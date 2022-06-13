#!/bin/bash
export test_data_dir="/global/cfs/cdirs/m2859/data/testdata"


export SPINIFEL_TEST_MODULE="ORIENTATION_MATCHING"
srun -n 1 python -u -m spinifel --settings=./settings/test_mpi.toml --mode=test


export SPINIFEL_TEST_MODULE="MAIN_PSANA2"
srun -n 3 python -u -m spinifel --settings=./settings/test_mpi.toml --mode=test
