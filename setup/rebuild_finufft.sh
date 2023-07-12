#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh

pushd "$root_dir"/finufft

cat > make.inc <<EOF
CFLAGS+=$(pkg-config --cflags fftw3)
LIBS+=$(pkg-config --libs-only-L fftw3)
EOF

make -j${THREADS:-8} lib
CFLAGS="$(pkg-config --cflags fftw3)" LDFLAGS="$(pkg-config --libs-only-L fftw3)" pip install --no-deps .

popd
