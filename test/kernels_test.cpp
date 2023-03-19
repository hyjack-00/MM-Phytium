#include <cstdio>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include <omp.h>

#include "microkernels.h"

using std::string;
using std::cout;
using std::endl;

typedef std::chrono::high_resolution_clock Clock;
#define Dur(start,end) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

#define FILE_OUTPUT false  // 是否输出从 stdout 到 文件
#define ANS_CHECK true  // 是否进行答案检查

string ouput_file = "../output/output.dat";
#if FILE_OUTPUT == true
    #define OS ofs
    std::ofstream ofs;
#else
    #define OS cout
#endif

// 输入矩阵的随机数范围
#define RAND_UB 100  // [LB, UB)
#define RAND_LB -100

template <typename T>
void print_mat(T *mat, int ni, int nj) {
    std::ofstream fout;
    fout.open("../output/matrix.dat");
    if (!fout.is_open()) {
        cout << "Error opening file for result" << endl;
        exit(1);
    }
    for (int i = 0; i < ni; i ++) {
        for (int j = 0; j < nj; j ++) {
            fout << mat[i*nj + j] << " ";
        }
        fout << endl;
    }
}
template <typename T>
void naive(T *A, T *B, T *C, size_t ni, size_t nj, size_t nk) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < ni; i ++) {
        for (size_t j = 0; j < nj; j ++) {
            T cij = 0;
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
            OS << "Ans Wrong at [" << i/nj << "," << i%nj << "]  mat:" << mat[i] << " ans:" << ans[i] << endl;
            flag = 0;
        }
    if (flag) OS << "Ans Correct" << endl;
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

#define Ti 128
#define Tj 128
#define Tk 128
#define MIN(x,y) (((x)<(y))?(x):(y))
void outer_kernel(int32_t *A, int32_t *B, int32_t *C,
                  size_t ni, size_t nj, size_t nk) {
    // packing...

    // #pragma omp parallel for
    for (size_t i0 = 0; i0 < ni; i0 += Ti) {
        size_t i1 = MIN(ni, i0+Ti);
        for (size_t j0 = 0; j0 < nj; j0 += Tj) {
            size_t j1 = MIN(nj, j0+Tj);
            for (size_t k0 = 0; k0 < nk; k0 += Tk) {
                size_t k1 = MIN(nk, k0+Tk);
                OS << "hint " << i0 << " " << j0 << " " << k0 << endl;
                mks32_0(A+i0*nk+k0, B+k0*nj+j0, C+i0*nj+j0, i1, j1, k1, nk, nj);
            }
        }
    }
}

int main() {
    const int input_loop = 2;
    const int compute_loop = 2;
    const int ni = 1024, nj = 1024, nk = 1024;

    int32_t *A = (int32_t *) malloc(sizeof(int32_t) * ni * nk);
    int32_t *B = (int32_t *) malloc(sizeof(int32_t) * nk * nj);
    int32_t *C = (int32_t *) malloc(sizeof(int32_t) * ni * nj);
    int32_t *D = (int32_t *) malloc(sizeof(int32_t) * ni * nj);
    // float32_t *A = (float32_t *) malloc(sizeof(float32_t) * ni * nk);
    // float32_t *B = (float32_t *) malloc(sizeof(float32_t) * nk * nj);
    // float32_t *C = (float32_t *) malloc(sizeof(float32_t) * ni * nj);

    OS << "Test start" << endl;
    OS << "Loop: " << input_loop << "x" << compute_loop
       << ", Size: i" << ni << " j" << nj << " k" << nk << endl;
    OS << "Enable C answer check? " << (bool)ANS_CHECK << endl;
    #if FILE_OUTPUT == true
    cout << "File output: " << ouput_file << endl; 
    #endif

    double total_time2 = 0;
    for (int input = 0; input < input_loop; input ++) {
        rand_mat_s32(A, ni * nk, 1234);
        rand_mat_s32(B, nk * nj, 5678);
        // rand_mat_f32(A, ni * nk, 1234);
        // rand_mat_f32(B, nk * nj, 5678);

        double total_time1 = 0;
        for (int compute = 0; compute < compute_loop; compute ++) {
            rand_mat_s32(C, ni * nj, 1357);
            auto start = Clock::now();

            outer_kernel(A, B, C, ni, nj, nk);

            auto end = Clock::now();
            double dur = Dur(start, end);
            dur /= 1000000.0;
            total_time1 += dur; 
            OS << "compute time: " << dur << " secs" << endl;
        }
        OS << "  avg time1: " << total_time1/compute_loop << " secs for input: " << input << endl;
        total_time2 += total_time1;

        // { // 对比
        //     rand_mat_s32(D, ni * nj, 1357);
        //     auto start = Clock::now();

        //     naive(A, B, D, ni, nj, nk);

        //     auto end = Clock::now();
        //     double dur = Dur(start, end);
        //     dur /= 1000000.0;
        //     OS << "  base time: " << dur << " secs" << endl;

        //     #if ANS_CHECK == true
        //         ans_check_s32(C, D, ni, nj);
        //     #endif
        // }
    }
    OS << "    total avg time2: " << total_time2/input_loop << " secs" << endl;

    // print_mat(C, ni, nj);
    free(A);
    free(B);
    free(C);

    char cur_time[50];
    time_t now = time(NULL);
    strftime(cur_time, 50, "%x %X", localtime(&now));
    OS << "Test Finished at " << cur_time << endl;
}
