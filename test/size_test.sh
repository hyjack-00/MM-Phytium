#!/bin/bash

# 设置替换的大小值
sizes=(4 8 16 32 64 128 256 512 768 1024)
# sizes=(5 6 7 8 9 10 11 12 13 14 15 16 20)
# sizes=(4 8 16 32 64 128)
# sizes=(256 512 768 1024)
# sizes=(2048 4096)

# 编译和运行的函数
compile_and_run() {
    sed -i "s/const int SIZE = 1024;/const int SIZE = $1;/g" kernels_eigen_test.cpp
    g++ -o eigen_test.exe kernels_eigen_test.cpp -I/usr/local/include/eigen3 -std=c++11 -O3 -march=native
    ./eigen_test.exe
    sed -i "s/const int SIZE = $1;/const int SIZE = 1024;/g" kernels_eigen_test.cpp
}

# 循环替换并编译运行
# 请先设置固定的 const int SIZE = 1024;
for size in "${sizes[@]}"
do
    echo "Testing with SIZE = $size"
    compile_and_run $size
    echo ""
done

