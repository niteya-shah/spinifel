#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
cd "$root_dir"

# Clean up any previous installs.
rm -rf conda

# Make the env.sh and unenv.sh
./mk_env.sh

source env.sh
# sometimes LD_PRELOAD can inverfere with other scripts here -- temporarily
# disable it (it is re-enabled at the end of this script)
__LD_PRELOAD=$LD_PRELOAD
unset LD_PRELOAD

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

CC=$MPI4PY_CC MPICC=$MPI4PY_MPICC pip install -v --no-binary mpi4py mpi4py

if [[ $USE_GASNET -eq 1 && $GASNET_ROOT == $PWD/gasnet/release ]]; then
    rm -rf gasnet
    git clone https://github.com/StanfordLegion/gasnet.git
    pushd gasnet
    make -j${THREADS:-8}
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
git clone -b 3.2.0-legion https://github.com/slac-lcls/lcls2.git $LCLS2_DIR
./psana_clean_build.sh

if [[ $(hostname) = *"tulip"* || $(hostname) = *"jlse"* ]]; then
    ./rebuild_fftw.sh
fi

rm -rf finufft
git clone https://github.com/elliottslaughter/finufft.git
./rebuild_finufft.sh

rm -rf pysingfel
git clone https://github.com/AntoineDujardin/pysingfel.git
./rebuild_pysingfel.sh

rm -rf cufinufft
git clone -b spinifel https://github.com/JBlaschke/cufinufft.git
./rebuild_cufinufft.sh

./mapper_clean_build.sh

pip install callmonitor

pip check

echo
echo "Done. Please run 'source env.sh' to use this build."

# Restore the LD_PRELOAD variable
export LD_PRELOAD=$__LD_PRELOAD
