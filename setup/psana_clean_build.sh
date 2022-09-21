#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh

pushd $LCLS2_DIR
git clean -fxd
popd

"$root_dir"/psana_dirty_build.sh
# FIXME: With setuptools pinned to 46.4.0, the current psana2 python packages
# will be installed into site-packages withou easy-install.pth file. This file
# points to the source folders where the modules are for 'development' mode.
# This is a hack to create the easy-install.pth file for psana2 and to be
# removed if we upgrade setuptools. Note!!! that newer setuptools breaks legion
# - see issue#48 for more detail.
cat << EOF > $LCLS2_DIR/install/lib/python$PYVER/site-packages/easy-install.pth
$LCLS2_DIR/psalg
$LCLS2_DIR/psana
EOF

