#!/bin/zsh

cd ./build
cmake ..
make
cd ..

export OMP_NUM_THREADS=4
export OMP_PROC_BIND=close
export OMP_PLACES=cores
./bin/kernels_test