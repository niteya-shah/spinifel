#!/usr/bin/env bash

set -e

# Ensure that the local conda environment has been loaded
root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh

# Enable host overwrite
target=${SPINIFEL_TARGET:-${NERSC_HOST:-$(hostname --fqdn)}}

# Get pybind11 configurations for the current system
pybind11_inclues=$(python3 -m pybind11 --includes)
pybind11_suffix=$(python3-config --extension-suffix)


#_______________________________________________________________________________
# Build the pybind11 CUDA KNN implmentation
#

pushd "${root_dir}/../spinifel/sequential/"

if [[ ${target} = "cgpu"* ]]
then
    nvcc -O3 -shared -std=c++11 --compiler-options -fPIC ${pybind11_inclues} \
        orientation_matching_sp.cu -o pyCudaKNearestNeighbors_SP${pybind11_suffix}
    nvcc -O3 -shared -std=c++11 --compiler-options -fPIC ${pybind11_inclues} \
        orientation_matching_dp.cu -o pyCudaKNearestNeighbors_DP${pybind11_suffix}
elif [[ ${target} = "perlmutter"* ]]
then
    nvcc -O3 -shared -std=c++11 --compiler-options -fPIC ${pybind11_inclues} \
        orientation_matching_sp.cu -o pyCudaKNearestNeighbors_SP${pybind11_suffix}
    nvcc -O3 -shared -std=c++11 --compiler-options -fPIC ${pybind11_inclues} \
        orientation_matching_dp.cu -o pyCudaKNearestNeighbors_DP${pybind11_suffix}
elif [[ ${target} = *"tulip"* || ${target} = *"jlse"* ]]
then
    nvcc -O3 -shared -std=c++11 --compiler-options -fPIC ${pybind11_inclues} \
        orientation_matching_sp.cu -o pyCudaKNearestNeighbors_SP${pybind11_suffix}
    nvcc -O3 -shared -std=c++11 --compiler-options -fPIC ${pybind11_inclues} \
        orientation_matching_dp.cu -o pyCudaKNearestNeighbors_DP${pybind11_suffix}
elif [[ ${target} = *"summit"* || ${target} = *"ascent"* ]]
then
    nvcc -O3 -shared -std=c++11 ${pybind11_inclues} \
        orientation_matching_sp.cu -o pyCudaKNearestNeighbors_SP${pybind11_suffix}
    nvcc -O3 -shared -std=c++11 ${pybind11_inclues} \
        orientation_matching_dp.cu -o pyCudaKNearestNeighbors_DP${pybind11_suffix}
elif [[ $(hostname --fqdn) = *".crusher."* || $(hostname --fqdn) = *".frontier."* ]]
then
    hipcc -O3 -shared -std=c++11 -fPIC --offload-arch=gfx90a -DUSE_HIP ${pybind11_inclues} \
        orientation_matching_sp.cu -o pyCudaKNearestNeighbors_SP${pybind11_suffix}
    hipcc -O3 -shared -std=c++11 -fPIC --offload-arch=gfx90a -DUSE_HIP ${pybind11_inclues} \
        orientation_matching_dp.cu -o pyCudaKNearestNeighbors_DP${pybind11_suffix}
elif [[ $(hostname --fqdn) = *".spock."* ]]
then
    hipcc -O3 -shared -std=c++11 -fPIC --amdgpu-target=gfx908 -DUSE_HIP ${pybind11_inclues} \
        orientation_matching_sp.cu -o pyCudaKNearestNeighbors_SP${pybind11_suffix}
    hipcc -O3 -shared -std=c++11 -fPIC --amdgpu-target=gfx908 -DUSE_HIP ${pybind11_inclues} \
        orientation_matching_dp.cu -o pyCudaKNearestNeighbors_DP${pybind11_suffix}
else
    echo "Don't recognize this target/hostname: ${target}."
    echo "Falling back to Intel-style system."
    nvcc -O3 -shared -std=c++11 --compiler-options -fPIC ${pybind11_inclues} \
        orientation_matching_sp.cu -o pyCudaKNearestNeighbors_SP${pybind11_suffix}
    nvcc -O3 -shared -std=c++11 --compiler-options -fPIC ${pybind11_inclues} \
        orientation_matching_dp.cu -o pyCudaKNearestNeighbors_DP${pybind11_suffix}
fi

popd

#-------------------------------------------------------------------------------
