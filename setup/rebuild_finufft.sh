#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh

pushd "$root_dir"/finufft

cat > make.inc <<EOF
CFLAGS+=$FINUFFT_CFLAGS
LIBS+=$FINUFFT_LDFLAGS
EOF

make -j8 lib
LDFLAGS="$FINUFFT_LDFLAGS" CFLAGS="$FINUFFT_CFLAGS" pip install --no-deps .

popd
