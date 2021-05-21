#!/bin/bash

set -e

fix_lib () {
    for so_file in $1/*.so
    do
        ln -sf $so_file
    done
}

if [[ -d $CONDA_PREFIX ]]
then
    pushd $CONDA_PREFIX/lib64
    fix_lib $OLCF_GCC_ROOT/lib64
    fix_lib /usr/lib64
    fix_lib /lib64
    popd
else
    echo "Could not find target conda library"
fi
