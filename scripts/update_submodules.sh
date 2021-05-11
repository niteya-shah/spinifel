#!/usr/bin/env bash


submodule_path () {
    echo $(echo $@ | cut -f 3 -d ' ')
}


submodule_hash () {
    echo $(echo $@ | cut -f 2 -d ' ')
}


update_submodules () {
    local IFS=$'\n'

    for sm in $(git submodule)
    do
        local git_path=$(submodule_path $sm)
        local git_hash=$(submodule_hash $sm)
        
        echo "Working on $sm"

        pushd $git_path
        git checkout $git_hash
        popd
    done
}


update_submodules
