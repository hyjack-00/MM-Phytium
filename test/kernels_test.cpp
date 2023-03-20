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

string ouput_file = "output/output.dat";
#if FILE_OUTPUT == true
    #define OS ofs
    std::ofstream ofs(ouput_file, std::ios::app);
#else
    #define OS cout
#endif

// 输入矩阵的随机数范围
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
template <typename T>
void zeros(T *mat, int length_1d) {
    for (int i = 0; i < length_1d; i ++) mat[i] = 0;
}


#define Ti 256
#define Tj 256
#define Tk 256
#define MIN(x,y) (((x)<(y))?(x):(y))

void outer_kernel_packAB(
    int32_t *A, int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    mks32_pAB_t mk, packs32_A_t pkA, packs32_B_t pkB) 
{
    int32_t *Apack = (int32_t *) malloc(sizeof(int32_t) * Ti * Tk);
    int32_t *Bpack = (int32_t *) malloc(sizeof(int32_t) * Tk * Tj);

    // #pragma omp parallel for collapse(2)
    for (size_t i0 = 0; i0 < ni; i0 += Ti) {
        for (size_t j0 = 0; j0 < nj; j0 += Tj) {
            size_t it = MIN(ni-i0, Ti);
            size_t jt = MIN(nj-j0, Tj);
            for (size_t k0 = 0; k0 < nk; k0 += Tk) {
                size_t kt = MIN(nk-k0, Tk);
                pkA(A+i0*nk+k0, Apack, it, kt, nk);
                pkB(B+k0*nj+j0, Bpack, kt, jt, nj);
                // print_mat(Apack, it*kt/32, 32, "Apack");
                // print_mat(Bpack, kt*jt/64, 64, "Bpack");
                mk(Apack, Bpack, C+i0*nj+j0, it, jt, kt, nj);
            }
        }
    }
}
void outer_kernel(
    int32_t *A, int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk, mks32_t mk) 
{
    // #pragma omp parallel for
    for (size_t i0 = 0; i0 < ni; i0 += Ti) {
        size_t it = MIN(ni-i0, Ti);
        for (size_t j0 = 0; j0 < nj; j0 += Tj) {
            size_t jt = MIN(nj-j0, Tj);
            for (size_t k0 = 0; k0 < nk; k0 += Tk) {
                size_t kt = MIN(nk-k0, Tk);
                mk(A+i0*nk+k0, B+k0*nj+j0, C+i0*nj+j0, it, jt, kt, nk, nj);
            }
        }
    }
}

#define TEST_N 1024

int main() {
    const int input_loop = 10;
    const int compute_loop = 10;
    // const int ni = 4, nj = 8, nk = 8;
    const int ni = TEST_N, nj = TEST_N, nk = TEST_N;

    OS << "Test start" << endl;
    OS << "Loop: " << input_loop << "x" << compute_loop
       << ", Size: i" << ni << " j" << nj << " k" << nk << endl;
    #if FILE_OUTPUT == true
    cout << "File output: " << ouput_file << endl; 
    #endif

    double total_time2 = 0;
    for (int input = 0; input < input_loop; input ++) {
        int32_t *A = (int32_t *) malloc(sizeof(int32_t) * ni * nk);
        int32_t *B = (int32_t *) malloc(sizeof(int32_t) * nk * nj);
        int32_t *C = (int32_t *) malloc(sizeof(int32_t) * ni * nj);
        int32_t *D = (int32_t *) malloc(sizeof(int32_t) * ni * nj);
        rand_mat_s32(A, ni * nk, 1234);
        rand_mat_s32(B, nk * nj, 5678);
        // rand_mat_s32(A, ni * nk, time(0));
        // rand_mat_s32(B, nk * nj, time(0)+1);

        double total_time1 = 0;
        for (int compute = 0; compute < compute_loop; compute ++) {
            zeros(C, ni * nj);
            auto start = Clock::now();

            // outer_kernel(A, B, C, ni, nj, nk, mks32_4x8k8_ldA_fchC);
            outer_kernel_packAB(A, B, C, ni, nj, nk, 
                mks32_4x8k8_ldB_fchC_pkAB, 
                packs32_4x8k8_A,
                packs32_4x8k8_B);

            auto end = Clock::now();
            double dur = Dur(start, end);
            dur /= 1000.0;
            total_time1 += dur; 
            OS << "compute time: " << dur << " msecs" << endl;
        }
        OS << "  avg time1: " << total_time1/compute_loop << " msecs for input: " << input << endl;
        total_time2 += total_time1/compute_loop;

        { // 对比
            zeros(D, ni * nj);
            auto start = Clock::now();

            outer_kernel(A, B, D, ni, nj, nk, mks32_0);

            auto end = Clock::now();
            double dur = Dur(start, end);
            dur /= 1000.0;
            OS << "  base time: " << dur << " msecs" << endl;

            ans_check_s32(C, D, ni, nj);
        }

        // print_mat(A, ni, nk, "A");
        // print_mat(B, nk, nj, "B");
        // print_mat(C, ni, nj, "C");
        // print_mat(D, ni, nj, "D");
        free(A);
        free(B);
        free(C);
        free(D);
    }
    OS << "    total avg time2: " << total_time2/input_loop << " msecs" << endl;

    char cur_time[50];
    time_t now = time(NULL);
    strftime(cur_time, 50, "%x %X", localtime(&now));
    OS << "Test Finished at " << cur_time << endl;

    #if FILE_OUTPUT == true
        ofs.close();
    #endif
}
