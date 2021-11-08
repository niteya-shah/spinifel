#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh
rm -rf "$root_dir"/finufft
git clone -b oneapi-support https://github.com/servesh/finufft.git

pushd "$root_dir"/finufft

ln -s make.inc.linux_ICX make.inc
make -j${THREADS:-8} lib
pip install --no-cache-dir --no-deps --use-feature=in-tree-build .

popd
