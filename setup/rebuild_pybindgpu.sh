#!/usr/bin/env bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh

# Enable host overwrite
target=${SPINIFEL_TARGET:-${NERSC_HOST:-$(hostname --fqdn)}}
echo $target

# ***TMP*** Checking out specific branch from a forked gitsubmodule
pushd "$root_dir"/PybindGPU
git checkout crusher
popd

pushd "$root_dir"/PybindGPU/PyGPU

if [[ ${target} = *"crusher"* ]]; then
    make 
elif [[ ${target} = *"spock"* ]]; then
    echo "Currently unsupported"
    exit
else
    echo "We do not need PybindGPU for this architecture"
    exit
fi
popd
pushd "$root_dir"/PybindGPU
if [[ ${target} = *"crusher"* ]]; then
    pip install --no-cache-dir -e .
fi
popd
