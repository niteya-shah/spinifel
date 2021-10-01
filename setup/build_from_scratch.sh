#!/bin/bash


set -e


root_dir=$(readlink -f $(dirname "${BASH_SOURCE[0]}"))
pushd $root_dir

# Enable host overwrite
target=${SPINIFEL_TARGET:-${NERSC_HOST:-$(hostname --fqdn)}}

# Enable CUDA build
cuda_build=${SPINIFEL_BUILD_CUDA:-true}
# except on certain targets
if [[ ${target} = *"tulip"* || ${target} = *"jlse"* ]]
then
    cuda_build=false
fi


#_______________________________________________________________________________
# Make clean environment

rm -rf conda
rm -rf install
mkdir -p install/bin
mkdir -p install/include
mkdir -p install/lib
mkdir -p install/share

#-------------------------------------------------------------------------------


#_______________________________________________________________________________
# Set up environment

# Make the env.sh and unenv.sh
./mk_env.sh

source env.sh

# sometimes LD_PRELOAD can inverfere with other scripts here -- temporarily
# disable it (it is re-enabled at the end of this script)
__LD_PRELOAD=$LD_PRELOAD
unset LD_PRELOAD

#-------------------------------------------------------------------------------


#_______________________________________________________________________________
# Install conda and set up local environment

# Clean up any previous installs.

# Get and run the miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -p).sh -O conda-installer.sh
bash ./conda-installer.sh -b -p $CONDA_ROOT
rm conda-installer.sh

source $CONDA_ROOT/etc/profile.d/conda.sh

PACKAGE_LIST=(
    python=$PYVER
    matplotlib
    numpy
    scipy
    pytest
    h5py

    cffi  # Legion
    pybind11  # FINUFFT
    numba  # skopi 
    scikit-learn  # skopi
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

# Extra deps required for psana machines, since there is no module system
if [[ ${target} = "psbuild"* ]]; then
    conda install -y compilers openmpi cudatoolkit-dev -c conda-forge
fi

# Install pip packages
CC=$MPI4PY_CC MPICC=$MPI4PY_MPICC pip install -v --no-binary mpi4py mpi4py
pip install --no-cache-dir callmonitor
pip install --no-cache-dir PyNVTX
pip install --no-cache-dir mrcfile
pip install --no-cache-dir cupy

#-------------------------------------------------------------------------------


#_______________________________________________________________________________
# Overwrite the conda libraries with system libraries => don't let anaconda
# provide libraries (like openmp) that are already provided by the system

if [[ ${target} = *"summit"* || ${target} = *"ascent"* ]]
then
    ${root_dir}/../scripts/fix_lib_olcf.sh
fi

#-------------------------------------------------------------------------------

#_______________________________________________________________________________
# Install UCX

if [[ $GASNET_CONDUIT = "ucx" ]]
then
    ./rebuild_ucx.sh
fi

#-------------------------------------------------------------------------------


#_______________________________________________________________________________
# Install GASNET

if [[ $LEGION_USE_GASNET -eq 1 && $GASNET_ROOT == ${root_dir}/gasnet/release ]]
then
    pushd gasnet
    CONDUIT=$GASNET_CONDUIT make -j${THREADS:-8}
    popd
fi

#-------------------------------------------------------------------------------


#_______________________________________________________________________________
# Install Legion

if [[ $LG_RT_DIR == ${root_dir}/legion/runtime ]]
then
    ./reconfigure_legion.sh
    ./rebuild_legion.sh
    ./mapper_clean_build.sh
fi

#-------------------------------------------------------------------------------


#_______________________________________________________________________________
# Install LCLS2 (aka PSANA2)

./psana_clean_build.sh

#-------------------------------------------------------------------------------


#_______________________________________________________________________________
# Rebuild FFTW (only needed on some systems -- that don't supply their own)

if [[ ${target} = *"tulip"* || ${target} = *"jlse"* || ${target} = "g0"*".stanford.edu" || ${target} = "psbuild"* ]]; then
    ./rebuild_fftw.sh
fi

#-------------------------------------------------------------------------------


#_______________________________________________________________________________
# Install finufft

./rebuild_finufft.sh

#-------------------------------------------------------------------------------


#_______________________________________________________________________________
# Install skopi (formerly known as pysingfel)

./rebuild_skopi.sh

#-------------------------------------------------------------------------------


#_______________________________________________________________________________
# Install cufinufft

if [[ ${cuda_build} == true ]]
then
    ./rebuild_cufinufft.sh
fi

#-------------------------------------------------------------------------------


#_______________________________________________________________________________
# Install CUDA KNN implmentation

if [[ ${cuda_build} == true ]]
then
    ./rebuild_knn.sh
fi

#-------------------------------------------------------------------------------





# pip check # FIXME (Elliott): this seems to be failing


echo
echo "Done. Please run 'source setup/env.sh' to use this build."

# Restore the LD_PRELOAD variable
export LD_PRELOAD=$__LD_PRELOAD
export popd
