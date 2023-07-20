#!/bin/bash

set -e

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root_dir"/../legion_mapper

source "$root_dir"/env.sh

mkdir -p build
cd build
cmake ..
make -j${THREADS:-8}
