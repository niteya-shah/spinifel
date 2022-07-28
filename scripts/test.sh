#!/bin/bash

set -e

root_dir="$PWD"
echo "root_dir: $root_dir"

source "$root_dir"/setup/env.sh

export PYTHONPATH="$PYTHONPATH:$root_dir"

set -x

# Pickup all tests in root and spinifel modules
pytest -s tests
pytest -s spinifel/tests

# Pickup tests as specified in init file in root/tests
# See documentation on Wiki page.
"${root_dir}"/tests/run.sh
