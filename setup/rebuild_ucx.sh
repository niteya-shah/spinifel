#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh

pushd "$root_dir"/ucx

# compiler wrappers on Cray machines are broken and don't support all linker flags required by UCX
unset CC
unset CXX

./autogen.sh
./contrib/configure-release --prefix=$LEGION_INSTALL_DIR
make -j${THREADS:-8}
make install

popd
