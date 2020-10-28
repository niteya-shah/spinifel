#!/usr/bin/env bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh

pushd "$root_dir"/cufinufft

if [[ $(hostname) = "cgpu"* ]]; then
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUFINUFFT_DIR make -j8 site=nersc_cgpu lib
elif [[ $(hostname --fqdn) = *"summit"* ]]; then
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUFINUFFT_DIR make -j8 site=olcf_summit lib
else
    echo "Cannot build cuFINUFFT for this architecture"
    exit
fi
make python
popd
