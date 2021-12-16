#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh

pushd "$root_dir"/finufft

cat > make.inc <<EOF
CFLAGS+=$FINUFFT_CFLAGS
CXXFLAGS+=$FINUFFT_CFLAGS
LIBS+=$FINUFFT_LDFLAGS
EOF

make -j${THREADS:-8} lib
pushd python
LDFLAGS="$FINUFFT_LDFLAGS" CFLAGS="$FINUFFT_CFLAGS" pip install --no-deps --use-feature=in-tree-build .
popd

popd
