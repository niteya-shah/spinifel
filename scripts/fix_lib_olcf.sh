#!/bin/bash

set -e

fix_lib () {

    __fix_lib_link () {

        blacklist=(openjp2 libuv) # ssl crypto krb5 stdc++
        fn=$(basename -- $1)

        safe=true
        for bl in ${blacklist[@]}
        do
            if [[ $fn == *${bl}* ]]
            then
                safe=false
                break
            fi
        done

        if [[ -e $fn ]] && [[ $safe == true ]]
        then
            echo "Overwriting: $fn with $1" >> fix_lib.log
            mv $fn fix_lib_moveasie/
            ln -sf $1
        else
            echo "Skipping: $fn due to blacklist" >> fix_lib.log
        fi
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
