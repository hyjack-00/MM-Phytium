#include <iostream>
#include <armpl.h>

#include <cstdio>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include <omp.h>
#include <arm_neon.h>

#include "test_helpers.h"

using namespace std;

#define MIN(x,y) (((x)<(y))?(x):(y))

#define TEST_N_F32 1024
int TEST_Ni_F32 = 1024;
int TEST_Nj_F32 = 1024;
int TEST_Nk_F32 = 1024;

const int data_loop = 5;
const int warmup_loop = 2;
const int compute_loop = 100;

void test_s32() {
    const int ni = TEST_Ni_F32, nj = TEST_Nj_F32, nk = TEST_Nk_F32;
    cout << "Size: i" << ni << " j" << nj << " k" << nk << endl;

    double total_time = 0;
    for (int data_i = 0; data_i < data_loop; data_i ++) {
        int *A = (int *) malloc(sizeof(int) * ni * nk);
        int *B = (int *) malloc(sizeof(int) * nk * nj);
        int *C = (int *) malloc(sizeof(int) * ni * nj);
        rand_mat_s32(A, ni * nk, 1234);
        rand_mat_s32(B, nk * nj, 5678);
        int alpha = 1, beta = 0;

        for (int i = 0; i < warmup_loop; i ++) {
            cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ni, nj, nk, &alpha, A, nk, B, nj, &beta, C, nj);
        }
        zeros_s32(C, ni * nj);

        double time = 0;
        auto start = Clock::now();
        for (int compute_i = 0; compute_i < compute_loop; compute_i ++) {

            cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ni, nj, nk, &alpha, A, nk, B, nj, &beta, C, nj);
        
        }
        auto end = Clock::now();
        double dur = Dur(start, end);
        dur /= 1000.0;
        time += dur; 
        total_time += time / compute_loop;

        free(A);
        free(B);
        free(C);
    }
    double total_avg_time = total_time / data_loop;
    double gflops = (double) ni * nj * nk * 2 / total_avg_time / 1e6;
    cout << "    total avg time: " << total_avg_time << " msecs" << endl;
    cout << "    " << gflops << " GOPS" << endl;  
    // GFLOPS = fop / 1e9 / secs = fop / (msecs / 1e3 * 1e9)  
}

void test_f32() {
    const int ni = TEST_Ni_F32, nj = TEST_Nj_F32, nk = TEST_Nk_F32;
    cout << "Size: i" << ni << " j" << nj << " k" << nk << endl;

    double total_time = 0;
    for (int data_i = 0; data_i < data_loop; data_i ++) {
        float32_t *A = (float32_t *) malloc(sizeof(float32_t) * ni * nk);
        float32_t *B = (float32_t *) malloc(sizeof(float32_t) * nk * nj);
        float32_t *C = (float32_t *) malloc(sizeof(float32_t) * ni * nj);
        rand_mat_f32(A, ni * nk, 1234);
        rand_mat_f32(B, nk * nj, 5678);

        for (int i = 0; i < warmup_loop; i ++) {
            // 执行 GEMM
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ni, nj, nk, 1.0, A, nk, B, nj, 0.0, C, nj);
        }
        zeros_f32(C, ni * nj);

        double time = 0;
        auto start = Clock::now();
        for (int compute_i = 0; compute_i < compute_loop; compute_i ++) {

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ni, nj, nk, 1.0, A, nk, B, nj, 0.0, C, nj);
        
        }
        auto end = Clock::now();
        double dur = Dur(start, end);
        dur /= 1000.0;
        time += dur; 
        total_time += time / compute_loop;

        free(A);
        free(B);
        free(C);
    }
    double total_avg_time = total_time / data_loop;
    double gflops = (double) ni * nj * nk * 2 / total_avg_time / 1e6;
    const double peak = 17.6; // 18.4
    cout << "    total avg time: " << total_avg_time << " msecs" << endl;
    cout << "    " << gflops << " GFLOPS" << endl;  
    // GFLOPS = fop / 1e9 / secs = fop / (msecs / 1e3 * 1e9)  
}

void test_f64() {
    const int ni = TEST_Ni_F32, nj = TEST_Nj_F32, nk = TEST_Nk_F32;
    cout << "Size: i" << ni << " j" << nj << " k" << nk << endl;

    double total_time = 0;
    for (int data_i = 0; data_i < data_loop; data_i ++) {
        double *A = (double *) malloc(sizeof(double) * ni * nk);
        double *B = (double *) malloc(sizeof(double) * nk * nj);
        double *C = (double *) malloc(sizeof(double) * ni * nj);
        rand_mat_f64(A, ni * nk, 1234);
        rand_mat_f64(B, nk * nj, 5678);

        for (int i = 0; i < warmup_loop; i ++) {
            // 执行 GEMM
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ni, nj, nk, 1.0, A, nk, B, nj, 0.0, C, nj);
        }

        zeros_f64(C, ni * nj);

        double time = 0;
        auto start = Clock::now();
        for (int compute_i = 0; compute_i < compute_loop; compute_i ++) {

            // 执行 GEMM
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ni, nj, nk, 1.0, A, nk, B, nj, 0.0, C, nj);

        }
        auto end = Clock::now();
        double dur = Dur(start, end);
        dur /= 1000.0;
        time += dur; 
        total_time += time / compute_loop;

        free(A);
        free(B);
        free(C);
    }
    double total_avg_time = total_time / data_loop;
    double gflops = (double) ni * nj * nk * 2 / total_avg_time / 1e6;
    const double peak = 8.8; // 9.2
    cout << "    total avg time: " << total_avg_time << " msecs" << endl;
    cout << "    " << gflops << " GFLOPS" << endl;  
    // GFLOPS = fop / 1e9 / secs = fop / (msecs / 1e3 * 1e9)  
}

int main() {
    // vector<int> sizes = {4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 2048, 4096};
    vector<int> sizes = {5,6,7,8,9,10,11,12,13,14,15,16,20};
    // for (auto size : sizes) {
    //     TEST_Ni_F32 = size;
    //     TEST_Nj_F32 = size;
    //     TEST_Nk_F32 = size;
    //     test_f32();
    // }
    // cout << endl;
    for (auto size : sizes) {
        TEST_Ni_F32 = size;
        TEST_Nj_F32 = size;
        TEST_Nk_F32 = size;
        test_f64();
    }
    // for (auto size : sizes) {
    //     TEST_Ni_F32 = size;
    //     TEST_Nj_F32 = size;
    //     TEST_Nk_F32 = size;
    //     test_s32();
    // }
}