#!/bin/zsh

cd ./build
cmake ..
make
cd ..

echo "cleared from ./run.sh" > output/matrix.dat

export OMP_NUM_THREADS=4
export OMP_PROC_BIND=close
export OMP_PLACES=cores
./bin/kernels_test