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
if [[ ${target} = "cori"* ]]; then
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

export LEGION_USE_GASNET=${LEGION_USE_GASNET:-1}
export GASNET_CONDUIT=aries
EOF
elif [[ ${target} = "cgpu"* ]]; then
    cat >> env.sh <<EOF
module purge
module load cgpu gcc cuda openmpi fftw python

export CC=gcc
export CXX=g++
# compilers for mpi4py
export MPI4PY_CC=gcc
export MPI4PY_MPICC=\$(which mpicc)

export LEGION_USE_GASNET=${LEGION_USE_GASNET:-1}
# NOTE: not sure if this is the best choice -- investigate further if this
# becomes a problem elsewhere
export GASNET_CONDUIT=ibv
export CROSS_CONFIGURE=
EOF
elif [[ ${target} = "perlmutter" ]]; then
    cat >> env.sh <<EOF
module load PrgEnv-gnu
module load gcc/9 # GCC 10 not supported by CUDA
module load cray-fftw

export CC=cc
export CXX=CC
export CRAYPE_LINK_TYPE=dynamic # allow dynamic linking

# compilers for mpi4py
export MPI4PY_CC="$(which cc)"
export MPI4PY_MPICC="$(which cc) --shared"

# Make sure Cray-FFTW get loaded first to avoid Conda's MKL
export LD_PRELOAD="\$FFTW_DIR/libfftw3.so"

export LEGION_USE_GASNET=${LEGION_USE_GASNET:-1}
export GASNET_CONDUIT=ucx
EOF
elif [[ ${target} = *"summit"* ]]; then
    cat >> env.sh <<EOF
module load gcc fftw cuda gsl

export CC=gcc
export CXX=g++
# compilers for mpi4py
export MPI4PY_CC=\$OMPI_CC
export MPI4PY_MPICC=mpicc

export LEGION_USE_GASNET=${LEGION_USE_GASNET:-1}
export GASNET_CONDUIT=ibv

# for Numba
export CUDA_HOME=\$OLCF_CUDA_ROOT
EOF
elif [[ ${target} = *"ascent"* ]]; then
    cat >> env.sh <<EOF
module load gcc fftw cuda gsl

export CC=gcc
export CXX=g++
# compilers for mpi4py
export MPI4PY_CC=\$OMPI_CC
export MPI4PY_MPICC=mpicc

export LEGION_USE_GASNET=${LEGION_USE_GASNET:-1}
export GASNET_CONDUIT=ibv

# for Numba
export CUDA_HOME=\$OLCF_CUDA_ROOT
EOF
elif [[ $(hostname -f) = *"alcf.anl.gov" ]]; then # arcticus
    cat >> env.sh <<EOF
module use /soft/restricted/CNDA/modulefiles
module load oneapi/release/latest # Get latest OneAPI release version
module load cmake # Get CMake with OneAPI support

export CONDA_AUTO_ACTIVATE_BASE=false

source \${IDPROOT}/bin/activate # Get Intel Python Env
export CC=icx
export CXX=icpx

# compilers for mpi4py
export MPI4PY_CC=icx
export MPI4PY_MPICC=mpicc

export LEGION_USE_GASNET=${LEGION_USE_GASNET:-0} # FIXME: GASNet on iris is currently broken
export GASNET_CONDUIT=ibv
EOF
elif [[ ${target} = *"tulip"* ]]; then
    cat >> env.sh <<EOF
# load a ROCm-compatible MPI
module use /home/groups/coegroup/share/coe/modulefiles
module load ompi/4.1.0/llvm/rocm/4.1.0

export CC=gcc
export CXX=g++

# compilers for mpi4py
export MPI4PY_CC=gcc
export MPI4PY_MPICC=mpicc

export LEGION_USE_GASNET=${LEGION_USE_GASNET:-1}
export GASNET_CONDUIT=ibv
EOF
elif [[ ${target} = "g0"*".stanford.edu" ]]; then # sapling
    cat >> env.sh <<EOF
module load cuda mpi slurm

export CC=gcc
export CXX=g++
# compilers for mpi4py
export MPI4PY_CC=gcc
export MPI4PY_MPICC=\$(which mpicc)

export LEGION_USE_GASNET=${LEGION_USE_GASNET:-1}
export GASNET_CONDUIT=ibv
EOF
elif [[ ${target} = "psbuild"* ]]; then # psana machines
    cat >> env.sh <<EOF
export CC=gcc
export CXX=g++
# compilers for mpi4py
export MPI4PY_CC=gcc
export MPI4PY_MPICC=mpicc

export LEGION_USE_GASNET=${LEGION_USE_GASNET:-0}
EOF
elif [[ $(hostname --fqdn) = *".spock."* ]]; then
    cat >> env.sh <<EOF
module load wget
module load PrgEnv-gnu
module load rocm
module load cray-fftw

export CC=cc
export CXX=CC
export CRAYPE_LINK_TYPE=dynamic # allow dynamic linking

# compilers for mpi4py
export MPI4PY_CC="$(which cc)"
export MPI4PY_MPICC="$(which cc) --shared"

# Make sure Cray-FFTW get loaded first to avoid Conda's MKL
export LD_PRELOAD="\$FFTW_DIR/libfftw3.so"

export LEGION_USE_GASNET=${LEGION_USE_GASNET:-1}
export GASNET_CONDUIT=${GASNET_CONDUIT:-ucx}
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

export LEGION_INSTALL_DIR="${root_dir}/install"
pathappend \$LEGION_INSTALL_DIR/bin
ldpathappend \$LEGION_INSTALL_DIR/lib
pythonpathappend \$LEGION_INSTALL_DIR/lib/python\$PYVER/site-packages

export CONDA_ROOT="${root_dir}/conda"
export CONDA_ENV_DIR="\$CONDA_ROOT/envs/oneapi"

export LCLS2_DIR="${root_dir}/lcls2"

#finufft library dir
export FINUFFT_DIR="${root_dir}/finufft"
ldpathappend \$FINUFFT_DIR/lib

pathappend \$LCLS2_DIR/install/bin
pythonpathappend \$LCLS2_DIR/install/lib/python\$PYVER/site-packages

export CONDA_PKGS_DIRS="\$CONDA_ROOT/pkgs"
export CONDA_ENVS_PATH="\$CONDA_ROOT/conda-envs"

if [[ -d \$CONDA_ROOT ]]; then
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
