#!/usr/bin/env bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh

# Enable host overwrite
target=${SPINIFEL_TARGET:-${NERSC_HOST:-$(hostname --fqdn)}}
echo $target

pushd "$root_dir"/PybindGPU
if [[ ${target} = *"crusher"* || ${target} = *"frontier"* ]]; then
    PYBIND_GPU_TARGET=gfx90a pip install --no-cache-dir -e .
else
    echo "We do not need PybindGPU for this architecture"
    exit
fi
popd