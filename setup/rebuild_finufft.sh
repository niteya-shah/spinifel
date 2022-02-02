#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh

# Enable host overwrite
target=${SPINIFEL_TARGET:-${NERSC_HOST:-$(hostname --fqdn)}}

pushd "$root_dir"/finufft

if [[ ${target} = *"summit"* || ${target} = *"ascent"* ]]; then
    MARCH_FLAGS="-mcpu=native"
else
    MARCH_FLAGS="-march=native"
fi

if [[ $CXX = "clang"* ]]; then
    OPT_FLAGS=""
else
    OPT_FLAGS="-fcx-limited-range"
fi

cat > make.inc <<EOF
CC:=$CC
CXX:=$CXX
CFLAGS:=-fPIC -O3 -funroll-loops $MARCH_FLAGS $OPT_FLAGS $FINUFFT_CFLAGS
FFLAGS:=\$(CFLAGS)
CXXFLAGS:=\$(CFLAGS) -DNEED_EXTERN_C
LIBS+=$FINUFFT_LDFLAGS
ENABLE_SINGLE:=OFF
EOF

make -j${THREADS:-8} lib
pushd python
LDFLAGS="$FINUFFT_LDFLAGS" CFLAGS="$FINUFFT_CFLAGS" pip install --no-deps --use-feature=in-tree-build .
popd

popd
