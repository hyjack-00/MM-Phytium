cmake_minimum_required(VERSION 3.0.0)
project(MM_Phytium)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

message("ARCH: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "aarch64")

    # kernels_test
    aux_source_directory(./microkernels KERNELS)
    set(KERNELS_TEST ./test/kernels_test.cpp)
    add_executable(kernels_test ${KERNELS_TEST} ${KERNELS})
    target_include_directories(kernels_test PUBLIC ./microkernels)

endif()

# sparce_test
set(SPARCE_TEST ./test/sparce_test.cpp)
add_executable(sparce_test ${SPARCE_TEST})
target_include_directories(sparce_test PUBLIC ./sparce)

# sparce_test_basic
set(SPARCE_TEST_BASIC ./test/sparce_test_basic.cpp)
add_executable(sparce_test_basic ${SPARCE_TEST_BASIC})
target_include_directories(sparce_test_basic PUBLIC ./sparce)

# compile options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -falign-loops -march=native -ffast-math -fopenmp -Wall -Wextra -g")

# add_compile_options(-static)
# add_compile_options(-funroll-loops)

