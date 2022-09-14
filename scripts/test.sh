#!/bin/bash

set -e

root_dir="$PWD"
echo "root_dir: $root_dir"

source "$root_dir"/setup/env.sh

export PYTHONPATH="$PYTHONPATH:$root_dir"

set -x

# Pickup all tests spinifel modules (anything that don't import mpi)
#pytest -s tests
#pytest -s spinifel/tests

# Running pytest on scripts that potentially use mpi fails (originally 
# seen in psana2 branch - see issue#56) The solution right now is to wrap 
# it under jsrun/srun.
target=${SPINIFEL_TARGET:-${NERSC_HOST:-$(hostname --fqdn)}}
echo "[spinifel/tests/withmpi] target: $target"

if [[ ${target} = "cgpu"* ]]; then
    SPINIFEL_TEST_LAUNCHER="srun -n1 -G1"
elif [[ ${target} = *"summit"* || ${target} = *"ascent"* ]]; then
    SPINIFEL_TEST_LAUNCHER="jsrun -n1 -g1"
fi
    
$SPINIFEL_TEST_LAUNCHER pytest -s tests/withmpi
$SPINIFEL_TEST_LAUNCHER pytest -s spinifel/tests/withmpi

