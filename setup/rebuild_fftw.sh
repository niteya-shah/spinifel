#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh

pushd "$root_dir"

rm -f fftw-3.3.8.tar.gz
rm -rf fftw-3.3.8

wget http://www.fftw.org/fftw-3.3.8.tar.gz
tar xfz fftw-3.3.8.tar.gz

mkdir fftw-3.3.8/build
pushd fftw-3.3.8/build
../configure --enable-openmp --disable-mpi --with-pic --prefix=$CONDA_ENV_DIR
make install -j8
popd # fftw-3.3.8/build

popd # "$root_dir"
