#!/usr/bin/env bash

set -e

# Ensure that the local conda environment has been loaded
root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh

# Enable host overwrite
target=${SPINIFEL_TARGET:-${NERSC_HOST:-$(hostname --fqdn)}}


#_______________________________________________________________________________
# Build the pybind11 CUDA device managment implmentation
#

pushd "${root_dir}/../spinifel/device/"

make all

popd