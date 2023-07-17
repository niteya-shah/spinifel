#!/bin/bash

set -e

target=${SPINIFEL_TARGET:-$(hostname --fqdn)}

root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh

source "$root_dir"/legion_build_dir.sh

pushd "$legion_build"

make -j${THREADS:-8}
make install

popd
