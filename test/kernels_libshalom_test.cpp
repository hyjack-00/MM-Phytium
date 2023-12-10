#include <cstdio>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <queue>

#include <omp.h>

#include "LibShalom.h"

#define MIN(x,y) (((x)<(y))?(x):(y))

#define TEST_N_F32 1024
int TEST_Ni_F32 = 1024;
int TEST_Nj_F32 = 1024;
int TEST_Nk_F32 = 1024;

void test_f32() {
    const int data_loop = 3;
    const int warmup_loop = 2;
    const int compute_loop = 3;
    // const int ni = 4, nj = 8, nk = 8;
    // const int ni = TEST_N_F32, nj = TEST_N_F32, nk = TEST_N_F32;
    const int ni = TEST_Ni_F32, nj = TEST_Nj_F32, nk = TEST_Nk_F32;

    cout << "Loop: " << data_loop << "x" << compute_loop << endl;
    cout << "Size: i" << ni << " j" << nj << " k" << nk << endl;

    double total_time = 0;
    for (int data_i = 0; data_i < data_loop; data_i ++) {
        float32_t *A = (float32_t *) malloc(sizeof(float32_t) * ni * nk);
        float32_t *B = (float32_t *) malloc(sizeof(float32_t) * nk * nj);
        float32_t *C = (float32_t *) malloc(sizeof(float32_t) * ni * nj);
        rand_mat_f32(A, ni * nk, 1234);
        rand_mat_f32(B, nk * nj, 5678);

        for (int i = 0; i < warmup_loop; i ++) {
            LibShalom_sgemm(NoTrans, NoTrans, C, A, B, ni, nj, nk);
        }

        double time = 0;
        for (int compute_i = 0; compute_i < compute_loop; compute_i ++) {
            zeros_f32(C, ni * nj);
            auto start = Clock::now();

            /* Timing Zone -- */

            LibShalom_sgemm(NoTrans, NoTrans, C, A, B, ni, nj, nk);

            /* -- Timing Zone */

            auto end = Clock::now();
            double dur = Dur(start, end);
            dur /= 1000.0;
            time += dur; 
            // cout << "compute time: " << dur << " msecs" << endl;
        }
        // cout << "  avg time: " << time/compute_loop << " msecs for data: " << data_i << endl;
        total_time += time/compute_loop;

        // // Answer Check, commented if not used
        //     float32_t *D = (float32_t *) malloc(sizeof(float32_t) * ni * nj);
        //     zeros_f32(D, ni * nj);
        //     auto start = Clock::now();
        //     naive(A, B, D, ni, nj, nk);
        //     auto end = Clock::now();
        //     double dur = Dur(start, end);
        //     cout << "naive time: " << dur / 1e6 << " secs" << endl;
        //     ans_check_f32(C, D, ni, nj); 
        //     free(D);

        free(A);
        free(B);
        free(C);
    }
    double total_avg_time = total_time / data_loop;
    double gflops = (double) ni * nj * nk * 2 / total_avg_time / 1e6;
    const double peak = 17.6; // 18.4
    cout << "    total avg time: " << total_avg_time << " msecs" << endl;
    cout << "    " << gflops << "GFLOPS, " << gflops * 100 / peak << "\% peak" << endl;  
    // GFLOPS = fop / 1e9 / secs = fop / (msecs / 1e3 * 1e9)  
}

int main() {
    vector<int> sizes = {4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 2048, 4096};
    for (auto size : sizes) {
        TEST_Ni_F32 = size;
        TEST_Nj_F32 = size;
        TEST_Nk_F32 = size;
        test_f32();
    }
}