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
elif [[ ${target} = "psbuild"* ]]; then
    export NVCCFLAGS="-std=c++17 -ccbin=${CXX} -O3 ${NVARCH} -Wno-deprecated-gpu-targets --default-stream per-thread -Xcompiler "\"${CXXFLAGS}\"
    make -j${THREADS:-8} site=psbuild lib
elif [[ ${target} = *"crusher"* || ${target} = *"frontier"* ]]; then
    make -j${THREADS:-8} site=olcf_crusher 
elif [[ ${target} = "g0"*".stanford.edu" ]]; then # sapling
    export NVCCFLAGS="-std=c++17 -ccbin=${CXX} -O3 ${NVARCH} -Wno-deprecated-gpu-targets --compiler-options=-fPIC --default-stream per-thread -Xcompiler "\"${CXXFLAGS}\"
    make -j${THREADS:-8}
elif [[ ${target} = "darwin"* ]]; then
    export NVCCFLAGS="-std=c++17 -ccbin=${CXX} -O3 ${NVARCH} -Wno-deprecated-gpu-targets --compiler-options=-fPIC --default-stream per-thread -Xcompiler "\"${CXXFLAGS}\"
    make -j${THREADS:-8}
else
    echo "Cannot build cuFINUFFT for this architecture"
    exit
fi

LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUFINUFFT_DIR" pip install --no-deps --no-cache-dir .

popd
