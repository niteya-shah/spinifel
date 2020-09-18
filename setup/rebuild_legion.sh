#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh

source "$root_dir"/legion_build_dir.sh

pushd "$legion_build"

make -j8
make install

popd
