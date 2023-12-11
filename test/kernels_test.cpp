#include <cstdio>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <queue>

#include <omp.h>

#include "test_helpers.h"
#include "microkernels.h"

#define MIN(x,y) (((x)<(y))?(x):(y))


static size_t Ti = 256;
static size_t Tj = 256;
static size_t Tk = 256;

void kernel_s32_packAB(
    int32_t *A, int32_t *B, int32_t *C,
    size_t ni, size_t nj, size_t nk,
    mks32_pAB_t mk, packs32_A_t pkA, packs32_B_t pkB) 
{
    int32_t *Apack = (int32_t *) malloc(sizeof(int32_t) * Ti * Tk);
    int32_t *Bpack = (int32_t *) malloc(sizeof(int32_t) * Tk * Tj);

    for (size_t i0 = 0; i0 < ni; i0 += Ti) {
        size_t it = MIN(ni-i0, Ti);
        for (size_t k0 = 0; k0 < nk; k0 += Tk) {
            size_t kt = MIN(nk-k0, Tk);
            for (size_t j0 = 0; j0 < nj; j0 += Tj) {
                size_t jt = MIN(nj-j0, Tj);
            
                pkA(A+i0*nk+k0, Apack, it, kt, nk);
                pkB(B+k0*nj+j0, Bpack, kt, jt, nj);
                // print_mat(Apack, it*kt/32, 32, "Apack");
                // print_mat(Bpack, kt*jt/64, 64, "Bpack");
                mk(Apack, Bpack, C+i0*nj+j0, it, jt, kt, nj);
            }
        }
    }
}

void kernel_f32_packAB(
    float32_t *A, float32_t *B, float32_t *C,
    size_t ni, size_t nj, size_t nk,
    mkf32_pAB_t mk, packf32_A_t pkA, packf32_B_t pkB) 
{
    float32_t *Apack = (float32_t *) malloc(sizeof(float32_t) * Ti * Tk);
    float32_t *Bpack = (float32_t *) malloc(sizeof(float32_t) * Tk * Tj);

    for (size_t i0 = 0; i0 < ni; i0 += Ti) {
        size_t it = MIN(ni-i0, Ti);
        for (size_t k0 = 0; k0 < nk; k0 += Tk) {
            size_t kt = MIN(nk-k0, Tk);
            for (size_t j0 = 0; j0 < nj; j0 += Tj) {
                size_t jt = MIN(nj-j0, Tj);
            
                pkA(A+i0*nk+k0, Apack, it, kt, nk);
                pkB(B+k0*nj+j0, Bpack, kt, jt, nj);
                mk(Apack, Bpack, C+i0*nj+j0, it, jt, kt, nj);
            }
        }
    }
}

//             // kernel_s32(A, B, C, ni, nj, nk, mks32_4x8k8_ldB_fchC);
//             // kernel_s32_packAB(A, B, C, ni, nj, nk, 
//             //     mks32_4x8k8_ldB_fchC_pkAB,
//             //     packs32_4x8k8_A,
//             //     packs32_4x8k8_B);
//             // kernel_packABC_s32(A, B, C, ni, nj, nk, 
//             //     mks32_4x8k8_ldB_fchC_pkABC, 
//             //     packs32_4x8k8_A,
//             //     packs32_4x8k8_B,
//             //     packs32_4x8k8_C,
//             //     unpacks32_4x8k8_C);


#define TEST_N_F32 1024
int TEST_Ni_F32 = 1024;
int TEST_Nj_F32 = 1024;
int TEST_Nk_F32 = 1024;

void test_f32() {
    const int data_loop = 3;
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

        double time = 0;
        for (int compute_i = 0; compute_i < compute_loop; compute_i ++) {
            zeros_f32(C, ni * nj);
            auto start = Clock::now();

            /* Timing Zone -- */

            // SMM_kernel_f32_single(C, A, B, ni, nj, nk, nj);
            // kernel_f32_packAB(A, B, C, ni, nj, nk, 
            //     mkf32_4x8k8_ldB_fchC_pkAB,
            //     packf32_4x8k8_A,
            //     packf32_4x8k8_B);

            /* -- Timing Zone */

            auto end = Clock::now();
            double dur = Dur(start, end);
            dur /= 1000.0;
            time += dur; 
            // cout << "compute time: " << dur << " msecs" << endl;
        }
        // cout << "  avg time: " << time/compute_loop << " msecs for data: " << data_i << endl;
        total_time += time/compute_loop;

        // Answer Check, commented if not used
            float32_t *D = (float32_t *) malloc(sizeof(float32_t) * ni * nj);
            zeros_f32(D, ni * nj);
            auto start = Clock::now();
            naive(A, B, D, ni, nj, nk);
            auto end = Clock::now();
            double dur = Dur(start, end);
            cout << "naive time: " << dur / 1e6 << " secs" << endl;
            ans_check_f32(C, D, ni, nj); 
            free(D);

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
    // test_s32();
    for (int size = 16; size <= 4096; size *= 2) {
        TEST_Ni_F32 = size;
        TEST_Nj_F32 = size;
        TEST_Nk_F32 = size;
        test_f32();
    }
}