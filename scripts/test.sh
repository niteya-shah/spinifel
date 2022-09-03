#!/bin/bash

set -e

root_dir="$PWD"
echo "root_dir: $root_dir"

source "$root_dir"/setup/env.sh

export PYTHONPATH="$PYTHONPATH:$root_dir"

set -x

# Pickup all tests spinifel modules
#pytest -s tests            # This will not work [my guess: pytest/spinifel settings conflict]
                            # - see issue#50 for more detail.
pytest -s tests/test_FSC.py

# Pickup tests as specified in init file in root/tests
# See documentation on Wiki page.
"${root_dir}"/tests/run.sh
