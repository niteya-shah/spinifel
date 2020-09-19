#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
cd "$root_dir"

# Setup environment.
if [[ $(hostname) = "cori"* ]]; then
    cat > env.sh <<EOF
module swap PrgEnv-intel PrgEnv-gnu
module load cray-fftw

export CC=cc
export CXX=CC
export CRAYPE_LINK_TYPE=dynamic # allow dynamic linking

# disable Cori-specific Python environment
unset PYTHONSTARTUP

# Make sure Cray-FFTW get loaded first to avoid Conda's MKL
export LD_PRELOAD="\$FFTW_DIR/libfftw3.so"

export USE_CUDA=${USE_CUDA:-0}
export USE_OPENMP=${USE_OPENMP:-1}
export USE_GASNET=${USE_GASNET:-1}
export CONDUIT=${CONDUIT:-aries}
EOF
elif [[ $(hostname --fqdn) = *"summit"* ]]; then
    cat > env.sh <<EOF
module load gcc/9.1.0
module load fftw/3.3.8
module load cuda/9.2.148
module load gsl
export CC=gcc
export CXX=g++

export USE_CUDA=${USE_CUDA:-0}
export USE_OPENMP=${USE_OPENMP:-1}
export USE_GASNET=${USE_GASNET:-1}
export CONDUIT=${CONDUIT:-ibv}

# for Numba
export CUDA_HOME=\$OLCF_CUDA_ROOT
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
export PATH="\$LEGION_INSTALL_DIR/bin:\$PATH"
export LD_LIBRARY_PATH="\$LEGION_INSTALL_DIR/lib:\$LD_LIBRARY_PATH"
export PYTHONPATH="\$LEGION_INSTALL_DIR/lib/python\$PYVER/site-packages:\$PYTHONPATH"

export CONDA_ROOT="$PWD/conda"
export CONDA_ENV_DIR="\$CONDA_ROOT/envs/myenv"

export LCLS2_DIR="$PWD/lcls2"

export PATH="\$LCLS2_DIR/install/bin:\$PATH"
export PYTHONPATH="\$LCLS2_DIR/install/lib/python\$PYVER/site-packages:\$PYTHONPATH"

if [[ -d \$CONDA_ROOT ]]; then
  source "\$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "\$CONDA_ENV_DIR"
fi
EOF

# Clean up any previous installs.
rm -rf conda

source env.sh

# Install Conda environment.
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -p).sh -O conda-installer.sh
bash ./conda-installer.sh -b -p $CONDA_ROOT
rm conda-installer.sh
source $CONDA_ROOT/etc/profile.d/conda.sh

# conda install -y conda-build # Must be installed in root environment
PACKAGE_LIST=(
    python=$PYVER
    matplotlib
    numpy
    scipy
    pytest
    h5py

    cffi  # Legion
    pybind11  # FINUFFT
    numba  # pysingfel
    scikit-learn  # pysingfel
    tqdm  # convenience

    # lcls2
    setuptools=46.4.0  # temp need specific version
    cmake
    cython
    mongodb
    pymongo
    curl
    rapidjson
    ipython
    requests
    mypy
    prometheus_client
)

conda create -y -p "$CONDA_ENV_DIR" "${PACKAGE_LIST[@]}" -c defaults -c anaconda
conda activate "$CONDA_ENV_DIR"
# Extra lcls2 deps
conda install -y amityping -c lcls-ii
conda install -y bitstruct krtc -c conda-forge

if [[ $(hostname) = "cori"* ]]; then
    CC=gcc MPICC=cc pip install -v --no-binary mpi4py mpi4py
elif [[ $(hostname --fqdn) = *"summit"* ]]; then
    CC=$OMPI_CC MPICC=mpicc pip install -v --no-binary mpi4py mpi4py
fi

if [[ $USE_GASNET -eq 1 && $GASNET_ROOT == $PWD/gasnet/release ]]; then
    rm -rf gasnet
    git clone https://github.com/StanfordLegion/gasnet.git
    pushd gasnet
    make -j8
    popd
fi

if [[ $LG_RT_DIR == $PWD/legion/runtime ]]; then
    rm -rf legion
    rm -rf install
    git clone -b control_replication https://gitlab.com/StanfordLegion/legion.git
    ./reconfigure_legion.sh
    ./rebuild_legion.sh
    cp "$CONDA_ENV_DIR"/lib/libhdf5* "$LEGION_INSTALL_DIR"/lib/
fi

rm -rf lcls2
git clone https://github.com/slac-lcls/lcls2.git $LCLS2_DIR
./psana_clean_build.sh

rm -rf finufft
git clone https://github.com/elliottslaughter/finufft.git
./rebuild_finufft.sh

rm -rf pysingfel
git clone git@github.com:AntoineDujardin/pysingfel.git
./rebuild_pysingfel.sh

pip check

echo
echo "Done. Please run 'source env.sh' to use this build."
