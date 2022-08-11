#!/bin/bash


set -xe

export CI_PIPELINE_ID=000
export SPINIFEL_TEST_LAUNCHER="jsrun -n1 -a1 -g1"

./scripts/test.sh

$SPINIFEL_TEST_LAUNCHER python -m spinifel --default-settings=summit_ci.toml --mode=mpi runtime.use_cufinufft=false

PYTHONPATH="$PYTHONPATH:$EXTERNAL_WORKDIR:$PWD/mpi4py_poison_wrapper" $SPINIFEL_TEST_LAUNCHER legion_python -ll:py 1 -ll:csize 8192 legion_main.py --default-settings=summit_ci.toml --mode=legion
