#!/usr/bin/env bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh

# Enable host overwrite
target=${SPINIFEL_TARGET:-${NERSC_HOST:-$(hostname --fqdn)}}

pushd "$root_dir"/cufinufft

if [[ ${target} = "cgpu"* ]]; then
    make -j${THREADS:-8} site=nersc_cgpu lib
elif [[ ${target} = "perlmutter"* ]]; then
    make -j${THREADS:-8} site=nersc_cgpu lib
elif [[ ${target} = *"summit"* || ${target} = *"ascent"* ]]; then
    make -j${THREADS:-8} site=olcf_summit lib
else
    echo "Cannot build cuFINUFFT for this architecture"
    exit
fi
pip install --no-cache-dir pycuda
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUFINUFFT_DIR pip install --no-cache-dir .
popd
