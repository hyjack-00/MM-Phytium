cmake_minimum_required(VERSION 3.0.0)
project(MM_Phytium)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# kernels_test
aux_source_directory(./microkernels KERNELS)
set(TEST ./test/kernels_test.cpp)
add_executable(kernels_test ${TEST} ${KERNELS})
target_include_directories(kernels_test PUBLIC ./microkernels)

# compile options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -Wall -Wextra -g -falign-loops -march=native -ffast-math -fopenmp")

# add_compile_options(-static)
# add_compile_options(-funroll-loops)

