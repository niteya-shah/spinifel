#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh

./autogen.sh
./contrib/configure-release --prefix=$LEGION_INSTALL_DIR
make -j${THREADS:-8}
make install
