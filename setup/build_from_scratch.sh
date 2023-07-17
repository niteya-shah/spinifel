#!/bin/bash


set -e


if (return 0 2>/dev/null); then
    echo "Please do not source this script. That is:"
    echo
    echo "DO NOT:"
    echo "    source setup/build_from_scratch.sh"
    echo
    echo "DO:"
    echo "    ./setup/build_from_scratch.sh"
    echo
    echo "Sourcing causes many issues, so we don't support it."
    echo "Once this script is done, it will produce a file env.sh"
    echo "that you can source like:"
    echo
    echo "    source env.sh"
    return 0
fi


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

rm -rf conda # Note: we don't use conda any more, but clean it up if it exists
rm -rf spack
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

# sometimes LD_PRELOAD can interfere with other scripts here -- temporarily
# disable it (it is re-enabled at the end of this script)
__LD_PRELOAD="$LD_PRELOAD"
unset LD_PRELOAD

#-------------------------------------------------------------------------------


#_______________________________________________________________________________
# Install spack and set up local environment

git clone -c feature.manyFiles=true -b cupy-rocm https://github.com/eugeneswalker/spack.git $SPACK_ROOT
# git -C $SPACK_ROOT checkout 829b4fe8feeed7baa1a41127f08a15a7eabc8e20 # develop from 2023-07-14
source $SPACK_ROOT/share/spack/setup-env.sh

# Link the desired machine config
if [[ -z $SPACK_TARGET_MACHINE ]]; then
    echo "Please modify mk_env.sh to set SPACK_TARGET_MACHINE for this machine"
    exit 1
fi
ln -sf ./machines/$SPACK_TARGET_MACHINE.yaml ./spack_machine_config.yaml

# Set up local build cache
if [[ -z $SPACK_BUILD_CACHE ]]; then
    echo "Please modify mk_env.sh to set SPACK_BUILD_CACHE for this machine"
    exit 1
fi
if mkdir $SPACK_BUILD_CACHE 2>/dev/null; then
    # Make sure the group always has access to this directory.
    chmod g+rwxs $SPACK_BUILD_CACHE
fi

spack mirror add local_build_cache $SPACK_BUILD_CACHE
spack buildcache keys -it # FIXME (Elliott): do we still need this?

# Important: Spack can only install out of a build cache if the path is LONGER
# than where it was originally built. This option asks Spack to pad the
# install directory so that we avoid issues in the future.
spack config add config:install_tree:padded_length:512

cp ./machines/$SPACK_TARGET_MACHINE.lock ./spack.lock || true
spack -e . concretize ${SPACK_FORCE_CONCRETIZE:+-f -U}
cp ./spack.lock ./machines/$SPACK_TARGET_MACHINE.lock

spack -e . install --no-check-signature -j${THREADS:-8}

spack -e . buildcache push --unsigned --allow-root --update-index $SPACK_BUILD_CACHE $(spack find --format /{hash})

spack env activate .

# Install pip packages
pip install --no-cache-dir gdown

#-------------------------------------------------------------------------------


#_______________________________________________________________________________
# Install GASNET

if [[ $LEGION_USE_GASNET -eq 1 && $GASNET_ROOT == ${root_dir}/gasnet/release ]]
then
    pushd gasnet
    make -j${THREADS:-8}
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
# Install PybindGPU

if [[ ${cuda_build} == true ]]
then
    ./rebuild_pybindgpu.sh
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
# Install LCLS2 (aka PSANA2)

./psana_clean_build.sh

#-------------------------------------------------------------------------------


echo
echo "Done. Please run 'source setup/env.sh' to use this build."
