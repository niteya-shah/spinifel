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

set -x
which python
which pip
set +x
conda info --envs
conda list | grep pybind11
set -x
python -c 'import pybind11; print(pybind11.__file__)'
set +x

LDFLAGS="$FINUFFT_LDFLAGS" CFLAGS="$FINUFFT_CFLAGS" pip install --no-deps .

popd
