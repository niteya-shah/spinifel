#!/bin/bash

set -e

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root_dir"/../legion_mapper

rm -rf build
"$root_dir"/mapper_dirty_build.sh
