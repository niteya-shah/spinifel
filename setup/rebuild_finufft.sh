#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh

pushd "$root_dir"/finufft

make -j8 lib
pip install --no-deps .

popd
