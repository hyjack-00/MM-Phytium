#pragma once
#include <cstdio>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

using std::string;
using std::cout;
using std::endl;

typedef std::chrono::high_resolution_clock Clock;
#define Dur(start,end) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

// 矩阵数据的随机数范围
#define RAND_UB 10  // [LB, UB)
#define RAND_LB -10

template <typename T>
void print_mat(T *mat, int ni, int nj, string msg="") {
    std::ofstream fout("output/matrix.dat", std::ios::app);
    if (!fout.is_open()) {
        cout << "Error opening file for result" << endl;
        exit(1);
    }
    fout << msg << endl;
    for (int i = 0; i < ni; i ++) {
        for (int j = 0; j < nj; j ++) {
            fout << mat[i*nj + j] << " ";
        }
        fout << endl;
    }
    fout << endl;
    fout.close();
}

// 实现了 << 的数据类型
template <typename T>
void print_mat(T &mat, string msg="") {
    std::ofstream fout("output/matrix.dat", std::ios::app);
    if (!fout.is_open()) {
        cout << "Error opening file for result" << endl;
        exit(1);
    }
    fout << msg << endl;
    fout << mat;
    fout << endl;
    fout.close();
}

template <typename T>
void copy_mat(T *src, T *dst, int length_1d) {
    for (int i = 0; i < length_1d; i ++) dst[i] = src[i];
}

template <typename T>
void naive(T *A, T *B, T *C, size_t ni, size_t nj, size_t nk) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < ni; i ++) {
        for (size_t j = 0; j < nj; j ++) {
            T cij = C[i*nj + j];
            for (size_t k = 0; k < nk; k ++) {
                cij += A[i*nk + k] * B[k*nj + j];
            }
            C[i*nj + j] = cij;
        }
    }
}

void ans_check_s32(int *mat, int *ans, int ni, int nj) {
    int flag = 1;
    for (int i = 0; i < ni*nj; i ++)
        if (mat[i] != ans[i]) {
            cout << "Ans Wrong at [" << i/nj << "," << i%nj << "]  mat:" << mat[i] << " ans:" << ans[i] << endl;
            flag = 0;
        }
    if (flag) cout << "Ans Correct" << endl;
}

void zeros_s32(int *mat, int length_1d) {
    for (int i = 0; i < length_1d; i ++) mat[i] = 0;
}

void rand_mat_s32(int *mat, int length_1d, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < length_1d; i ++) 
        mat[i] = (rand() % (RAND_UB - RAND_LB)) + RAND_LB;
}

void rand_mat_f32(float *mat, int length_1d, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < length_1d; i ++) 
        mat[i] = ((float)rand() / ((float)RAND_MAX / (RAND_UB - RAND_LB))) + RAND_LB;
}


//## sparce matrix
// 使用一维数组存储的稀疏矩阵
void rand_sparce_array_s32(int *mat, int ni, int nj, double density, unsigned int seed) {
    srand(seed);
    zeros_s32(mat, ni*nj);
    int nnz = ni * nj * density;
    for (int elem = 0; elem < nnz; elem ++) {
        int i = rand() % ni;
        int j = rand() % nj;
        mat[i*nj + j] = (rand() % (RAND_UB - RAND_LB)) + RAND_LB;
    }
}
void rand_sparce_array_s32(int *mat, int length_1d, double density, unsigned int seed) {
    srand(seed);
    zeros_s32(mat, length_1d);
    int nnz = length_1d * density;
    for (int elem = 0; elem < nnz; elem ++) {
        int idx = rand() % length_1d;
        mat[idx] = (rand() % (RAND_UB - RAND_LB)) + RAND_LB;
    }
}