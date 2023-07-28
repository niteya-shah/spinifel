#!/bin/bash

set -e

target=${SPINIFEL_TARGET:-$(hostname --fqdn)}

root_dir="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
source "$root_dir"/env.sh

legion_build="$(mktemp -d)"
cat > "$root_dir"/legion_build_dir.sh <<EOF
legion_build="$legion_build"
EOF

pushd "$legion_build"

if [[ ${target} = "psbuild"* ]]; then
    export LDFLAGS="-Wl,-rpath,$CONDA_ENV_DIR/lib -lhdf5 -lz"
    alias cmake=${CONDA_PREFIX}/bin/cmake
else
    export LDFLAGS="-Wl,-rpath,$CONDA_ENV_DIR/lib"
fi

cmake -DCMAKE_PREFIX_PATH="$CONDA_ENV_DIR" \
    -DCMAKE_BUILD_TYPE=$([ $LEGION_DEBUG -eq 1 ] && echo Debug || echo Release) \
    -DBUILD_SHARED_LIBS=ON \
    -DLegion_BUILD_BINDINGS=ON \
    -DLegion_ENABLE_TLS=ON \
    -DLegion_USE_Python=ON \
    -DPYTHON_EXECUTABLE="$(which python)" \
    -DLegion_USE_CUDA=OFF \
    -DLegion_USE_OpenMP=ON \
    -DLegion_USE_GASNet=$([ $LEGION_USE_GASNET -eq 1 ] && echo ON || echo OFF) \
    -DGASNet_ROOT_DIR="$GASNET_ROOT" \
    -DGASNet_CONDUITS=${LEGION_GASNET_CONDUIT:-$GASNET_CONDUIT} \
    -DLegion_NETWORKS=gasnetex \
    -DLegion_USE_HDF5=ON \
    -DLegion_MAX_DIM=4 \
    -DCMAKE_INSTALL_PREFIX="$LEGION_INSTALL_DIR" \
    -DCMAKE_INSTALL_LIBDIR="$LEGION_INSTALL_DIR/lib" \
    "$root_dir"/legion

popd
