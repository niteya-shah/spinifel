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

# clear pip cache
rm -rf ~/.cache/pip

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
    setuptools
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


if [[ ${target} = "psbuild"* ]]; then
    conda create -y -p "$CONDA_ENV_DIR" "${PACKAGE_LIST[@]}" -c conda-forge
else
   conda create -y -p "$CONDA_ENV_DIR" "${PACKAGE_LIST[@]}" -c defaults -c anaconda
fi

conda activate "$CONDA_ENV_DIR"

# Extra lcls2 deps
conda install -y amityping -c lcls-ii
conda install -y bitstruct krtc -c conda-forge

# Important: install CuPy first, it is now a dependency for mpi4py (at least in some cases)
(
    if [[ $(hostname --fqdn) = *".crusher."* ]]; then
        export CUPY_INSTALL_USE_HIP=1
        export ROCM_HOME=$ROCM_PATH
        export HCC_AMDGPU_TARGET=gfx90a
        pip install --no-cache-dir cupy
    elif [[ $(hostname --fqdn) = *".spock."* ]]; then
        export CUPY_INSTALL_USE_HIP=1
        export ROCM_HOME=$ROCM_PATH
        export HCC_AMDGPU_TARGET=gfx908
        pip install --no-cache-dir cupy
    else
        pip install --no-cache-dir cupy
    fi
)

# Extra deps required for psana machines
if [[ ${target} = "psbuild"* ]]
then
    conda install -y -c conda-forge \
        compilers                   \
        openmpi                     \
        cudatoolkit=11.4            \
        cudatoolkit-dev=11.4        \
        cmake                       \
        make                        \
        cupy                        \
        mpi4py                      \
        mrcfile
else
    CC=$MPI4PY_CC MPICC=$MPI4PY_MPICC pip install -v --no-binary mpi4py mpi4py
    pip install --no-cache-dir mrcfile
    LDFLAGS=$CUPY_LDFLAGS pip install --no-cache-dir cupy
fi

# Install pip packages
pip install --no-cache-dir callmonitor
pip install --no-cache-dir PyNVTX

# Pin sckit-learn to 1.0.2 w/o breaking psana (see issue #51)
conda remove --force -y scikit-learn
conda install --freeze-installed -y scikit-learn=1.0.2

#-------------------------------------------------------------------------------


#_______________________________________________________________________________
# Install UCX

if [[ $GASNET_CONDUIT = "ucx" ]]
then
    ./rebuild_ucx.sh
    export GASNET_EXTRA_CONFIGURE_ARGS="--with-ucx-home=$LEGION_INSTALL_DIR --with-mpi-cc=$CC"
    export CROSS_CONFIGURE=
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


#_______________________________________________________________________________
# Overwrite the conda libraries with system libraries => don't let anaconda
# provide libraries (like openmp) that are already provided by the system

# FIXME (Elliott): After this point, CMake will be BROKEN. If you try
# to do any further builds, they will NOT work. The error looks like:
#
# cmake: symbol lookup error: cmake: undefined symbol: uv_fs_get_system_error
#
# This is probably happening because we're overwriting important
# libraries from Conda with system ones. Unfortunately, this does not
# work. I think the long term solution needs to be more precise about
# exactly what system libraries we're going to get (e.g., OpenMP, but
# not others). What we've got right now is far too indiscriminant. But
# as an immediate hack, we'll just put this as late in the build as
# possible so that we hope we don't mess with anything important.

#if [[ ${target} = *"summit"* || ${target} = *"ascent"* ]]
#then
#    ${root_dir}/../scripts/fix_lib_olcf.sh
#fi

#-------------------------------------------------------------------------------



# pip check # FIXME (Elliott): this seems to be failing


echo
echo "Done. Please run 'source setup/env.sh' to use this build."

# Restore the LD_PRELOAD variable
export LD_PRELOAD=$__LD_PRELOAD


#_______________________________________________________________________________
# Install LCLS2 (aka PSANA2)

./psana_clean_build.sh

#-------------------------------------------------------------------------------


export popd
