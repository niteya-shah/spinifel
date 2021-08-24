#!/bin/bash

set -e

fix_lib () {

    __fix_lib_link () {

        blacklist=(ssl crypto krb5 stdc++)
        fn=$(basename -- $1)

        for bl in ${blacklist[@]}
        do
            if [[ $fn == *${bl}* ]]
            then
                return 1
            fi
        done

        if [[ -e $fn ]]
        then
            echo "Overwriting: $fn with $1" >> fix_lib.log
            mv $fn fix_lib_moveasie/
        fi

        ln -sf $1

        return 0
    }

    mkdir -p fix_lib_moveasie

    for so_file in $1/*.so
    do
        __fix_lib_link $so_file
    done

    for so_file in $1/*.so.*
    do
        __fix_lib_link $so_file
    done
}



if [[ -d $CONDA_PREFIX ]]
then
    pushd $CONDA_PREFIX/lib
    fix_lib /usr/lib64
    fix_lib /lib64
    fix_lib $OLCF_GCC_ROOT/lib64
    popd
else
    echo "Could not find target conda library"
fi
