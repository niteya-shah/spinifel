#!/bin/bash

set -e

root_dir="$PWD"
echo "root_dir: $root_dir"

source "$root_dir"/setup/env.sh

export PYTHONPATH="$PYTHONPATH:$root_dir"

set -x

pytest spinifel/tests
# pytest tests
