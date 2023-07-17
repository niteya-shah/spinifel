#!/usr/bin/env bash

set -e

root_dir=$(readlink -f $(dirname "${BASH_SOURCE[0]}"))
pushd $root_dir

# Basics
cat > env.sh <<EOF
# Save environment before making changes to it:
export -p > $root_dir/saved-env.sh

# Append to PATH and LD_LIBRARY_PATH iff it's not already there
pathappend() {
    for ARG in "\$@"
    do
        if [[ -d "\$ARG" ]] && [[ ":\$PATH:" != *":\$ARG:"* ]]; then
            export PATH="\${PATH:+"\$PATH:"}\$ARG"
        fi
    done
}

ldpathappend() {
    for ARG in "\$@"
    do
        if [[ -d "\$ARG" ]] && [[ ":\$LD_LIBRARY_PATH:" != *":\$ARG:"* ]]; then
            export LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:+"\$LD_LIBRARY_PATH:"}\$ARG"
        fi
    done
}

pythonpathappend() {
    for ARG in "\$@"
    do
        if [[ -d "\$ARG" ]] && [[ ":\$PYTHONPATH:" != *":\$ARG:"* ]]; then
            export PYTHONPATH="\${PYTHONPATH:+"\$PYTHONPATH:"}\$ARG"
        fi
    done
}

_module_loaded () {
    mod_grep=\$(module list 2>&1 | grep \$@)

    if [[ -n \$mod_grep ]]; then
        return 0
    else
        return 1
    fi
}

EOF

# Enable host overwrite
target=${SPINIFEL_TARGET:-${NERSC_HOST:-$(hostname --fqdn)}}

# Setup environment.
if [[ ${target} = "perlmutter" ]]; then
    cat >> env.sh <<EOF
module load PrgEnv-gnu
module load gcc/11.2.0
module load cray-pmi # for GASNet
module load evp-patch # workaround for recent Perlmutter issue

export CC=cc
export CXX=CC
export CRAYPE_LINK_TYPE=dynamic # allow dynamic linking

export LEGION_USE_GASNET=${LEGION_USE_GASNET:-1}
export LEGION_GASNET_CONDUIT=ofi
export LEGION_GASNET_SYSTEM=slingshot11

export SPACK_BUILD_CACHE=/global/cfs/cdirs/m2859/spack_build_cache
export SPACK_TARGET_MACHINE=perlmutter
EOF
elif [[ ${target} = *"summit"* || ${target} = *"ascent"* ]]; then
    cat >> env.sh <<EOF
module load gcc/11.1.0

export CC=gcc
export CXX=g++

export LEGION_USE_GASNET=${LEGION_USE_GASNET:-1}
export LEGION_GASNET_CONDUIT=ibv

# for Numba
export CUDA_HOME=\$OLCF_CUDA_ROOT

export SPACK_BUILD_CACHE=/gpfs/wolf/chm137/proj-shared/spack_build_cache
export SPACK_TARGET_MACHINE=ascent
EOF
elif [[ ${target} = *"jlse"* ]]; then # iris, yarrow
    cat >> env.sh <<EOF
module load oneapi # just get some sort of a compiler loaded
module load mpi
export CC=icx
export CXX=icpx

export LEGION_USE_GASNET=${LEGION_USE_GASNET:-0} # FIXME: GASNet on iris is currently broken
export LEGION_GASNET_CONDUIT=ibv
EOF
elif [[ ${target} = "g0"*".stanford.edu" ]]; then # sapling
    cat >> env.sh <<EOF
module load cuda mpi slurm

export CC=gcc
export CXX=g++

export LEGION_USE_GASNET=${LEGION_USE_GASNET:-1}
export LEGION_GASNET_CONDUIT=ibv
EOF
elif [[ ${target} = "psbuild"* ]]; then # psana machines
    cat >> env.sh <<EOF
#export CC=gcc
#export CXX=g++

export LEGION_USE_GASNET=${LEGION_USE_GASNET:-0}
EOF
elif [[ $(hostname --fqdn) = *".frontier."* || $(hostname --fqdn) = *".crusher."* ]]; then
    cat >> env.sh <<EOF
module load PrgEnv-gnu
module load rocm/5.4.3

export CC=cc
export CXX=CC
export CRAYPE_LINK_TYPE=dynamic # allow dynamic linking

export LEGION_USE_GASNET=${LEGION_USE_GASNET:-1}
export LEGION_GASNET_CONDUIT=ofi
export LEGION_GASNET_SYSTEM=slingshot11

export SPACK_BUILD_CACHE=/lustre/orion/proj-shared/chm137/spack_build_cache
export SPACK_TARGET_MACHINE=frontier
EOF
elif [[ $(hostname --fqdn) = *"darwin"* ]]; then
    cat >> env.sh <<EOF
module load gcc
module load openmpi

export CC=gcc
export CXX=g++
EOF
else
    echo "I don't know how to build it on this machine..."
    exit 1
fi

cat >> env.sh <<EOF
export GASNET_ROOT="${GASNET_ROOT:-${root_dir}/gasnet/release}"

export LG_RT_DIR="${LG_RT_DIR:-${root_dir}/legion/runtime}"
export LEGION_DEBUG=0

export PYVER=3.8

export SPACK_ROOT="${root_dir}/spack"

# Do not look in either $HOME/.spack or any system files. This is to avoid
# issues where the user's preexiting configuration interferes with Spinifel.
export SPACK_DISABLE_LOCAL_CONFIG=1

export LOCAL_INSTALL_DIR="${root_dir}/install"

pathappend \${LOCAL_INSTALL_DIR}/bin
ldpathappend \${LOCAL_INSTALL_DIR}/lib
pythonpathappend \${LOCAL_INSTALL_DIR}/lib/python\${PYVER}/site-packages

export LCLS2_DIR="${root_dir}/lcls2"

#cufinufft library dir
export CUFINUFFT_DIR="${root_dir}/cufinufft/lib"
ldpathappend \$CUFINUFFT_DIR

pathappend \${LCLS2_DIR}/install/bin

pythonpathappend \${LCLS2_DIR}/install/lib/python\${PYVER}/site-packages

if [[ -d \$SPACK_ROOT ]]; then
  source "\${SPACK_ROOT}/share/spack/setup-env.sh"
  spack env activate "${root_dir}"
fi
EOF

# Build unenv script
cat > unenv.sh <<EOF
spack deactivate
if [[ -e $root_dir/saved-env.sh ]]; then
    for i in \$(env | awk -F"=" '{print \$1}') ; do unset \$i 2> /dev/null ; done
    source $root_dir/saved-env.sh
fi
EOF
popd
