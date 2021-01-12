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

# Setup environment.
if [[ $HOSTNAME = "cori"* ]]; then # On cori batch nodes, we don't have a usable $(hostname)
    cat >> env.sh <<EOF
if _module_loaded PrgEnv-intel; then
    module swap PrgEnv-intel PrgEnv-gnu
fi
module load cray-fftw

export CC=cc
export CXX=CC
export CRAYPE_LINK_TYPE=dynamic # allow dynamic linking

# compilers for mpi4py
export MPI4PY_CC="$(which cc)"
export MPI4PY_MPICC="$(which cc) --shared"

# disable Cori-specific Python environment
unset PYTHONSTARTUP

# Make sure Cray-FFTW get loaded first to avoid Conda's MKL
export LD_PRELOAD="\$FFTW_DIR/libfftw3.so"

export USE_CUDA=${USE_CUDA:-0}
export USE_OPENMP=${USE_OPENMP:-1}
export USE_GASNET=${USE_GASNET:-1}
export CONDUIT=${CONDUIT:-aries}
EOF
elif [[ $(hostname) = "cgpu"* ]]; then
    cat >> env.sh <<EOF
module purge
module load cgpu gcc cuda openmpi fftw python

export CC=gcc
export CXX=g++
# compilers for mpi4py
export MPI4PY_CC=gcc
export MPI4PY_MPICC=\$(which mpicc)

export USE_CUDA=${USE_CUDA:-1}
export USE_OPENMP=${USE_OPENMP:-1}
export USE_GASNET=${USE_GASNET:-1}
# NOTE: not sure if this is the best choice -- investigate further if this
# becomes a problem elsewhere
export CONDUIT=${CONDUIT:-ibv}
EOF
elif [[ $(hostname --fqdn) = *"summit"* || $(hostname --fqdn) = *"ascent"* ]]; then
    cat >> env.sh <<EOF
module load gcc/7.4.0
module load fftw/3.3.8
module load cuda/10.2.89
module load gsl
export CC=gcc
export CXX=g++

# compilers for mpi4py
export MPI4PY_CC=\$OMPI_CC
export MPI4PY_MPICC=mpicc

export USE_CUDA=${USE_CUDA:-0}
export USE_OPENMP=${USE_OPENMP:-1}
export USE_GASNET=${USE_GASNET:-1}
export CONDUIT=${CONDUIT:-ibv}

# for Numba
export CUDA_HOME=\$OLCF_CUDA_ROOT
EOF
elif [[ $(hostname) = *"jlse"* ]]; then # iris, yarrow
    cat >> env.sh <<EOF
module load oneapi # just get some sort of a compiler loaded
module load mpi
export CC=icx
export CXX=icpx

# compilers for mpi4py
export MPI4PY_CC=clang
export MPI4PY_MPICC=mpicc

export USE_CUDA=${USE_CUDA:-0}
export USE_OPENMP=${USE_OPENMP:-1}
export USE_GASNET=${USE_GASNET:-0} # FIXME: GASNet on iris is currently broken
export CONDUIT=${CONDUIT:-ibv}
EOF
elif [[ $(hostname) = *"tulip"* ]]; then
    cat >> env.sh <<EOF
# load a ROCm-compatible MPI
module use /home/users/twhite/share/modulefiles
module load ompi

export CC=gcc
export CXX=g++

# compilers for mpi4py
export MPI4PY_CC=gcc
export MPI4PY_MPICC=mpicc

export USE_CUDA=${USE_CUDA:-0}
export USE_OPENMP=${USE_OPENMP:-1}
export USE_GASNET=${USE_GASNET:-1}
export CONDUIT=${CONDUIT:-ibv}
EOF
else
    echo "I don't know how to build it on this machine..."
    exit 1
fi

cat >> env.sh <<EOF
export GASNET_ROOT="${GASNET_ROOT:-$PWD/gasnet/release}"

export LG_RT_DIR="${LG_RT_DIR:-$PWD/legion/runtime}"
export LEGION_DEBUG=1
export MAX_DIM=4
export USE_HDF=1

export PYVER=3.8

export LEGION_INSTALL_DIR="$PWD/install"
pathappend \$LEGION_INSTALL_DIR/bin
ldpathappend \$LEGION_INSTALL_DIR/lib
pythonpathappend \$LEGION_INSTALL_DIR/lib/python\$PYVER/site-packages

export CONDA_ROOT="$PWD/conda"
export CONDA_ENV_DIR="\$CONDA_ROOT/envs/myenv"

export LCLS2_DIR="$PWD/lcls2"

# settings for finufft
if [[ -z \${FFTW_INC+x} ]]; then
    export FINUFFT_CFLAGS="-I\$CONDA_ENV_DIR/include"
else
    export FINUFFT_CFLAGS="-I\$FFTW_INC -I\$CONDA_ENV_DIR/include"
fi
if [[ -z \${FFTW_DIR+x} ]]; then
    export FINUFFT_LDFLAGS="-L\$CONDA_ENV_DIR/lib"
else
    export FINUFFT_LDFLAGS="-L\$FFTW_DIR -L\$CONDA_ENV_DIR/lib"
fi

#cufinufft library dir
export CUFINUFFT_DIR="$PWD/cufinufft/lib"
ldpathappend \$CUFINUFFT_DIR

pathappend \$LCLS2_DIR/install/bin
pythonpathappend \$LCLS2_DIR/install/lib/python\$PYVER/site-packages

if [[ -d \$CONDA_ROOT ]]; then
  source "\$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "\$CONDA_ENV_DIR"
fi
EOF

# Build unenv script
cat > unenv.sh <<EOF
conda deactivate
if [[ -e $root_dir/saved-env.sh ]]; then
    for i in \$(env | awk -F"=" '{print \$1}') ; do unset \$i 2> /dev/null ; done
    source $root_dir/saved-env.sh
fi
EOF
popd
